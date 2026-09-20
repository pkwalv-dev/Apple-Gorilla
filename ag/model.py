"""Thin wrapper over the Anthropic Claude API.

Provides one `complete()` entry point used by every stage of the pipeline, plus
an offline `--dry-run` stub so the whole machine is demonstrable without a key.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import Config


@dataclass
class ModelResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stopped: str = "end_turn"
    dry_run: bool = False


class Cancelled(Exception):
    """Raised inside a streaming completion when its Canceller has been tripped."""


class Canceller:
    """A thread-safe stop signal for an in-flight generation.

    The generating thread registers its live HTTP response; another thread (the web
    server's /stop handler) calls cancel(), which trips the flag AND closes the
    response. Closing it unblocks a read that is stuck in the model's prompt-eval /
    reasoning phase, so a run can be interrupted promptly — not only once tokens flow.
    """

    def __init__(self) -> None:
        import threading
        self.stopped = False
        self._resp = None
        self._lock = threading.Lock()

    def register(self, resp) -> None:
        with self._lock:
            self._resp = resp

    def cancel(self) -> None:
        # Set the flag; the reading thread sees it between chunks and closes the
        # connection itself. We deliberately do NOT close the socket from this thread:
        # a cross-thread close of a blocked read is unreliable (notably on Windows) and
        # can leave the upstream generation running. The same-thread close on the next
        # chunk is what actually cancels the model.
        with self._lock:
            self.stopped = True


class DryRunClient:
    """Deterministic offline stand-in for the API.

    It echoes structured, plausible output so the pipeline (optimize/execute/
    critique/iterate/evolve) can be exercised and tested with no network or key.
    """

    def complete(self, *, system: str, user: str, cfg: Config,
                 max_tokens: Optional[int] = None, on_delta=None,
                 cancel=None) -> ModelResult:
        tag = _role_tag(system)
        if tag == "optimizer":
            text = (
                "SYSTEM:\nYou are a precise, well-calibrated assistant.\n\n"
                "USER:\n" + _first_line(user, "the request") +
                "\n\n(True engineered prompt is produced when an API key is set.)"
            )
        elif tag == "critic":
            text = json.dumps({
                "score": 9.0,
                "verdict": "pass",
                "issues": [],
                "fixes": [],
                "notes": "dry-run: critic stub auto-passes.",
            })
        elif tag == "ingest":
            text = (
                "# About Me\n## Who I am\n"
                "- (dry-run) set ANTHROPIC_API_KEY to distill a real profile\n"
                "## Domains I work in\n- (pending)\n"
                "## Standing preferences\n- (pending)\n"
            )
        elif tag == "evolver":
            text = json.dumps({
                "rationale": "dry-run: no change proposed offline.",
                "patches": [],
            })
        else:
            text = ("[dry-run answer] " + _first_line(user, "your prompt") +
                    "\nSet ANTHROPIC_API_KEY to get a real response.")
        if on_delta is not None:
            on_delta(text)   # exercise the streaming path (and interrupt hook) offline
        return ModelResult(text=text, model="dry-run", dry_run=True)


class ApiClient:
    """Live client. Uses adaptive thinking + effort per skill guidance."""

    def __init__(self) -> None:
        import anthropic  # imported lazily so dry-run needs no dependency
        self._anthropic = anthropic
        # Work around a decompression bug in some anthropic-SDK/httpx2 + zstandard
        # combos ("Decompressor.decompress() got an unexpected keyword argument
        # 'output_buffer_limit'"): ask the server not to compress, so the broken
        # decoder never runs. Responses here are small, so identity is fine.
        headers = {"Accept-Encoding": "identity"}
        kwargs = {"default_headers": headers, "timeout": 60.0, "max_retries": 1}
        # Credential resolution, in order: env var (SDK reads it) -> a key the user
        # saved through AG's own sign-in -> an OAuth profile (Claude subscription).
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            pass  # the SDK picks these up from the environment
        elif saved_api_key():
            kwargs["api_key"] = saved_api_key()
        else:
            # OAuth profile: the SDK does NOT auto-read that credentials file, so we
            # load the bearer token and pass it explicitly, plus the OAuth beta header
            # /v1/messages requires.
            token = _load_oauth_token()
            if token is None:
                raise RuntimeError(
                    "Not signed in to Claude. Sign in from AG (web app 'Sign in' box "
                    "or `ag login --key sk-...`), set ANTHROPIC_API_KEY, or use a "
                    "Claude Code / `ant auth login` OAuth profile."
                )
            headers["anthropic-beta"] = "oauth-2025-04-20"
            kwargs["auth_token"] = token
        self._client = anthropic.Anthropic(**kwargs)

    def complete(self, *, system: str, user: str, cfg: Config,
                 max_tokens: Optional[int] = None, on_delta=None,
                 cancel=None) -> ModelResult:
        mt = max_tokens or cfg.max_output_tokens
        # Stream for large generations to avoid HTTP timeouts (skill guidance).
        with self._client.messages.stream(
            model=cfg.model,
            max_tokens=mt,
            system=system,
            thinking=_anthropic_thinking(cfg),
            output_config={"effort": cfg.effort},
            messages=[{"role": "user", "content": user}],
        ) as stream:
            if cancel is not None:
                cancel.register(stream)          # /stop can close the stream
            if on_delta is not None or cancel is not None:
                # Forward answer text as it streams; a Stop (cancel/on_delta raising)
                # exits the context manager and halts the request.
                for chunk in stream.text_stream:
                    if cancel is not None and cancel.stopped:
                        raise Cancelled()
                    if chunk and on_delta is not None:
                        on_delta(chunk)
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        return ModelResult(
            text=text,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            model=msg.model,
            stopped=msg.stop_reason or "end_turn",
        )


class OllamaClient:
    """Local, keyless, offline backend via Ollama's HTTP API (stdlib only).

    Talks to `POST {host}/api/chat`. No account, no API key, no per-token cost.
    Quality tracks the local model you pull; the pipeline logic is identical.
    """

    def __init__(self, cfg: Config) -> None:
        self._host = cfg.ollama_host.rstrip("/")
        self._model = cfg.ollama_model

    def complete(self, *, system: str, user: str, cfg: Config,
                 max_tokens: Optional[int] = None, on_delta=None,
                 cancel=None) -> ModelResult:
        """Generate an answer. If `on_delta(text)` is given, stream the model's output
        live — every content/reasoning chunk is passed to it as it arrives. A `cancel`
        Canceller (if given) makes the call interruptible at any point: cancel() closes
        the Ollama connection, so even the prompt-eval/reasoning phase can be stopped."""
        import urllib.error
        import urllib.request

        # Merge tuned options (num_ctx, num_gpu, temperature, ...) from config; the
        # answer-length cap always wins. keep_alive keeps the model resident across
        # the pipeline's 3-4 calls per run, avoiding costly reloads (big latency win).
        options = dict(getattr(cfg, "ollama_options", {}) or {})
        options["num_predict"] = max_tokens or cfg.max_output_tokens
        # Extended-thinking OFF: Qwen3 (and similar) honour a `/no_think` switch in the
        # prompt — a cleaner suppression than Ollama's think=false (which can leak the
        # reasoning into the answer). Appended to the user turn.
        user_content = user + ("\n\n/no_think" if _no_think(cfg) else "")
        # Stream when the caller wants live tokens OR the ability to interrupt.
        stream = on_delta is not None or cancel is not None
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "stream": stream,
            "keep_alive": getattr(cfg, "ollama_keep_alive", "30m"),
            "options": options,
        }
        # `think` is a top-level Ollama field honoured by thinking models; only "on"
        # forces it. Non-thinking models reject it, so we retry once without it below.
        think = _ollama_think(cfg)
        if think is not None:
            payload["think"] = think

        def _open(body: dict):
            headers = {"Content-Type": "application/json"}
            if stream:
                # No keep-alive for streamed calls: we want closing the response to
                # actually tear down the socket so Ollama sees the disconnect and
                # cancels generation (keep-alive can otherwise hold it open).
                headers["Connection"] = "close"
            req = urllib.request.Request(
                f"{self._host}/api/chat",
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            return urllib.request.urlopen(req, timeout=600)

        try:
            try:
                resp = _open(payload)
            except urllib.error.HTTPError as he:
                body = ""
                try:
                    body = he.read().decode("utf-8", "replace")
                except Exception:
                    pass
                # Model doesn't support the think flag -> drop it and retry once.
                if "think" in payload and "think" in body.lower():
                    payload.pop("think", None)
                    resp = _open(payload)
                else:
                    raise
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._host} ({e}). "
                f"Is `ollama serve` running and `{self._model}` pulled?"
            ) from e

        if not stream:
            with resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = (data.get("message") or {}).get("content", "")
            pin = data.get("prompt_eval_count", 0) or 0
            pout = data.get("eval_count", 0) or 0
        else:
            content, pin, pout = self._consume_stream(resp, on_delta, cancel)

        text = _strip_thinking(content)
        # Defense in depth: if suppression/stripping left nothing but the model did
        # emit content (e.g. a truncated <think> with no answer after it), salvage the
        # de-tagged content rather than returning a blank answer to the pipeline.
        if not text and content.strip():
            text = re.sub(r"</?think>", "", content).strip()
        return ModelResult(text=text, input_tokens=pin, output_tokens=pout,
                           model=self._model)

    @staticmethod
    def _consume_stream(resp, on_delta, cancel=None):
        """Read Ollama's NDJSON stream, forwarding each chunk to on_delta live.

        Returns (full_content, prompt_tokens, eval_tokens). on_delta (if given) receives
        the model's reasoning (`thinking`) and answer (`content`) text as it streams and
        may raise to interrupt. A `cancel` Canceller is registered so /stop can close the
        connection mid-read; a read that fails after cancel() is reported as Cancelled.
        """
        parts, pin, pout = [], 0, 0
        if cancel is not None:
            cancel.register(resp)
        try:
            with resp:
                for raw in resp:
                    if cancel is not None and cancel.stopped:
                        raise Cancelled()
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    msg = obj.get("message") or {}
                    th = msg.get("thinking")
                    if th and on_delta is not None:
                        on_delta(th)                 # may raise to interrupt
                    c = msg.get("content")
                    if c:
                        parts.append(c)
                        if on_delta is not None:
                            on_delta(c)              # may raise to interrupt
                    if obj.get("done"):
                        pin = obj.get("prompt_eval_count", 0) or 0
                        pout = obj.get("eval_count", 0) or 0
        except (OSError, ValueError):
            # A closed-mid-read from cancel() surfaces here — report it as a cancel.
            if cancel is not None and cancel.stopped:
                raise Cancelled()
            raise
        return "".join(parts), pin, pout


def _ollama_reachable(host: str, timeout: float = 1.5) -> bool:
    """True if the Ollama HTTP API answers at `host` within `timeout` seconds."""
    import urllib.request
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 500
    except Exception:
        return False


def _is_local_host(host: str) -> bool:
    """True only for loopback/any-local hosts — we never try to start a remote server."""
    from urllib.parse import urlparse
    h = (urlparse(host).hostname or "").lower()
    return h in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "")


def _find_ollama_exe() -> Optional[str]:
    """Locate the `ollama` binary on PATH, or in common per-OS install locations."""
    import shutil
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidates = []
    if os.name == "nt":
        for base in (os.environ.get("LOCALAPPDATA"),
                     os.environ.get("ProgramFiles"),
                     os.environ.get("ProgramW6432")):
            if base:
                candidates.append(Path(base) / "Programs" / "Ollama" / "ollama.exe")
                candidates.append(Path(base) / "Ollama" / "ollama.exe")
    else:
        candidates += [
            Path("/usr/local/bin/ollama"), Path("/opt/homebrew/bin/ollama"),
            Path("/usr/bin/ollama"), Path.home() / ".local" / "bin" / "ollama",
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
        ]
    for c in candidates:
        try:
            if Path(c).exists():
                return str(c)
        except OSError:
            continue
    return None


def _spawn_ollama_serve(exe: str) -> bool:
    """Launch `ollama serve` detached so it outlives this process. Best-effort."""
    import subprocess
    try:
        if os.name == "nt":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: no console, survives us.
            flags = 0x00000008 | 0x00000200
            subprocess.Popen(
                [exe, "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True,
            )
        else:
            subprocess.Popen(
                [exe, "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        return True
    except Exception:
        return False


def ensure_ollama_running(cfg: Config, *, timeout: float = 12.0) -> bool:
    """Make sure the local Ollama server is up, starting it if needed.

    Best-effort and never raises: if Ollama can't be reached or launched, callers
    proceed unchanged and `OllamaClient.complete()` surfaces its own clear
    "is `ollama serve` running?" error. Returns True iff the API is reachable by the
    time we return.

    Only a *local* host is auto-started (we never poke a remote one), and only when
    `cfg.ollama_autostart` is set. After spawning, we poll until the HTTP API answers
    (the model itself loads lazily on first request — we only need the server up).
    """
    host = cfg.ollama_host
    if _ollama_reachable(host):
        return True
    if not getattr(cfg, "ollama_autostart", True) or not _is_local_host(host):
        return False
    exe = _find_ollama_exe()
    if not exe or not _spawn_ollama_serve(exe):
        return False
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _ollama_reachable(host, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def _ollama_think(cfg: Config):
    """Ollama's top-level `think` flag from cfg.think.

    Only "on" forces it. "off" is handled by a `/no_think` prompt suffix instead —
    Ollama's `think=false` leaks the model's reasoning into the answer on some builds
    rather than suppressing it. "auto" leaves the model to its default.
    """
    return True if getattr(cfg, "think", "auto") == "on" else None


def _no_think(cfg: Config) -> bool:
    """True when the user asked to turn extended thinking OFF.

    Only an explicit "off" suppresses thinking. "auto" means exactly what the UI
    says — leave the model to its own default (thinking-capable local models like
    Qwen3 will think). We never silently override the user's selection here; if a
    local thinking model is slow, that is surfaced in the UI's activity indicator and
    the user can pick a faster model or turn thinking off themselves."""
    return getattr(cfg, "think", "auto") == "off"


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove any chain-of-thought that leaked into a model's answer text.

    Thinking models emit reasoning in <think>…</think>. AG's answer should be the
    conclusion only, so drop paired blocks — and, defensively, an orphaned block that
    ends with </think> but lost its opener (seen with Ollama's think=false).
    """
    if not text:
        return text
    t = _THINK_RE.sub("", text)
    low = t.lower()
    if "</think>" in low and "<think>" not in low:
        t = t[low.rfind("</think>") + len("</think>"):]
    # Dangling opener with no closer: the reasoning was cut off (e.g. the token
    # budget ran out mid-think). Drop from the opener to the end — keeping any real
    # text that preceded it — so a truncated think block never leaks as the answer.
    low = t.lower()
    if "<think>" in low and "</think>" not in low:
        t = t[:low.find("<think>")]
    return t.strip()


def _anthropic_thinking(cfg: Config) -> dict:
    """Map cfg.think to the Claude API thinking config."""
    return ({"type": "disabled"} if getattr(cfg, "think", "auto") == "off"
            else {"type": "adaptive"})


def _has_anthropic_creds() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or saved_api_key())


def _saved_key_path() -> Path:
    """Where AG stores an API key the user saved via `ag login` / the web sign-in.

    Kept in the user's config dir (NOT the repo), so it is never committed and is
    shared with the OAuth profile location.
    """
    return _oauth_profile_dir() / "ag_api_key"


def saved_api_key() -> Optional[str]:
    """The API key the user saved through AG's own sign-in, if any."""
    try:
        key = _saved_key_path().read_text(encoding="utf-8").strip()
        return key or None
    except OSError:
        return None


def save_api_key(key: str) -> None:
    """Persist a user-provided API key locally (0600 where supported). The user
    enters their own key; AG only stores it so `auto` can use Claude."""
    key = (key or "").strip()
    if not key:
        raise ValueError("empty API key")
    p = _saved_key_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(key, encoding="utf-8")
    try:
        import os as _os
        _os.chmod(p, 0o600)
    except OSError:
        pass


def clear_api_key() -> bool:
    """Remove any saved API key (sign out of the key path). True if one existed."""
    try:
        _saved_key_path().unlink()
        return True
    except OSError:
        return False


def signin_status() -> dict:
    """How AG will authenticate to Claude right now — for the GUI/CLI sign-in view."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return {"signed_in": True, "method": "env"}
    if saved_api_key():
        return {"signed_in": True, "method": "api-key"}
    if has_oauth_profile():
        return {"signed_in": True, "method": "oauth", "token": oauth_token_status()}
    return {"signed_in": False, "method": "none"}


def _oauth_profile_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "anthropic"


def has_oauth_profile() -> bool:
    """True if an `ant auth login` OAuth profile appears to exist on disk.

    This is the sanctioned, key-free auth path. The SDK does NOT read this file
    itself, so ApiClient loads the token from it (see `_load_oauth_token`). Whether
    a given account/plan authorizes Messages API calls, and whether the token is
    still valid, is up to Anthropic — this only detects that a login happened.
    """
    creds = _oauth_profile_dir() / "credentials"
    try:
        # A real login writes credentials/<profile>.json. A bare config dir
        # (from `ant auth status`) must NOT count as authenticated.
        return creds.is_dir() and any(creds.glob("*.json"))
    except OSError:
        return False


def _read_oauth_cred() -> Optional[dict]:
    """Parse the first OAuth credentials JSON on disk (or None)."""
    creds = _oauth_profile_dir() / "credentials"
    try:
        for f in sorted(creds.glob("*.json")):
            try:
                return json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
    except OSError:
        pass
    return None


def _load_oauth_token() -> Optional[str]:
    """The OAuth access token to send as a bearer credential (or None)."""
    d = _read_oauth_cred() or {}
    tok = d.get("access_token")
    return tok if isinstance(tok, str) and tok else None


def oauth_token_status() -> str:
    """'none' | 'valid' | 'expired' | 'unknown' — for `ag doctor` to guide re-login."""
    d = _read_oauth_cred()
    if not d or not d.get("access_token"):
        return "none"
    exp = d.get("expires_at")
    if not exp:
        return "unknown"
    import time as _time
    exp_s = exp / 1000 if exp > 1e12 else exp  # tolerate ms or s epochs
    return "valid" if exp_s > _time.time() else "expired"


_OLLAMA_TAGS_CACHE: dict = {}   # retained: external callers may import it


def _ollama_tags(host: str, timeout: float) -> Optional[List[str]]:
    """The tag list for one host, TTL-cached.

    Cached per HOST, not per (host, name): one HTTP round trip answers every
    model question during routing instead of one probe per candidate. A TTL
    (not a permanent cache) means a model you pull — or delete — while AG is
    running is noticed within `PROBES.ttl`, which the old positive-only cache
    could never do. None means "the probe failed", which is deliberately NOT
    cached: a momentarily-down Ollama must not disable routing for the session.
    """
    from .cache import PROBES
    key = f"ollama_tags:{host}"
    hit = PROBES.get(key)
    if hit is not None:
        return list(hit)
    import json as _json
    import urllib.request
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=timeout) as r:
            tags = [m.get("name", "") for m in
                    _json.loads(r.read().decode("utf-8")).get("models", [])]
    except Exception:
        return None
    PROBES.put(key, list(tags))
    return tags


def ollama_has_model(cfg: Config, name: str, *, timeout: float = 1.5) -> bool:
    """Whether `name` is pulled locally, so routing can fall back gracefully instead of
    erroring mid-run on a model the user hasn't downloaded."""
    name = (name or "").strip()
    if not name:
        return False
    host = cfg.ollama_host.rstrip("/")
    tags = _ollama_tags(host, timeout)
    if tags is None:
        return False
    # Exact match, tolerating an implicit ":latest". A bare repo name (no tag) also
    # matches any pulled tag of that repo. NOT a loose prefix match: "qwen2.5:7b"
    # must not be reported as "qwen2.5:7b-instruct", or the call errors at run time.
    if ":" in name:
        return name in tags or (name + ":latest") in tags
    return any(t == name or t.split(":")[0] == name for t in tags)


def make_client(cfg: Optional[Config] = None, *, backend: Optional[str] = None,
                dry_run: bool = False):
    """Select a backend.

    Resolution: explicit `dry_run` wins; otherwise the given `backend` (or
    cfg.backend) decides. 'auto' uses Anthropic when creds exist, else dry-run.
    """
    cfg = cfg or Config()
    choice = "dry" if dry_run else (backend or cfg.backend or "auto")

    if choice == "dry":
        return DryRunClient()
    if choice == "ollama":
        ensure_ollama_running(cfg)
        return OllamaClient(cfg)
    if choice == "anthropic":
        return ApiClient()
    # auto: use Claude when an API key OR an OAuth profile is present; otherwise fall
    # back to the configured offline backend (default: local Ollama) so AG still
    # answers with a real model offline, never silently downgrading to the stub.
    if _has_anthropic_creds() or has_oauth_profile():
        try:
            return ApiClient()
        except Exception:
            pass
    offline = (getattr(cfg, "offline_backend", "") or "dry").lower()
    if offline == "ollama":
        ensure_ollama_running(cfg)
        return OllamaClient(cfg)
    return DryRunClient()


# --- helpers ---------------------------------------------------------------

def _role_tag(system: str) -> str:
    s = system.lower()
    for tag in ("optimizer", "critic", "evolver", "ingest"):
        if f"[role:{tag}]" in s:
            return tag
    return "executor"


def _first_line(text: str, default: str) -> str:
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line[:200] if line else default


def extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from a model reply (fenced or bare)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        candidate = m.group(1) if m else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# --- instruction-following compensation ------------------------------------- #
#
# Abliterated and merged models are deliberately run here (the specialist coder),
# and their known failure mode is DISCIPLINE, not knowledge: empty replies,
# repetition loops, prose-wrapped JSON, or ignoring an output contract. The harness
# cannot fix the weights, so it compensates at the call site: detect degenerate
# output, repair the prompt, retry, and fall back to a more reliable model. This
# is the runtime half of the instruction category in benchgen.py — that measures
# the weakness; this keeps a run working in spite of it.


def looks_degenerate(text: str) -> Optional[str]:
    """Detect output no user should ever see. Returns a reason string, or None.

    Deliberately narrow — a short or odd answer is fine; only patterns that are
    never legitimate qualify, so nothing interesting is ever second-guessed:

    - the reply is empty/whitespace;
    - the tail is one chunk repeated (a decoding repetition loop). The PERIOD is
      discovered from the text, not guessed: a 37-char loop must be caught just as
      surely as a 30- or 120-char one.
    """
    t = (text or "").strip()
    if not t:
        return "empty"
    tail = t[-600:]
    # Periodic check: is the tail just 3+ copies of a shorter unit? The period is
    # discovered, not guessed — a 37-char loop must be caught as surely as a neat
    # 120-char one. Right-anchored at the tail end, where a decoding loop shows.
    for period in range(24, len(tail) // 3 + 1):
        unit = tail[-period:]
        repeats = (unit * (len(tail) // period + 1))[-len(tail):]
        if repeats == tail:
            return f"repetition loop ({period}-char unit x{len(tail) // period}+)"
    return None


# Appended on a retry when a structured reply failed to parse. Plain imperative
# phrasing, and it goes AFTER the task: recency is what a weak instruction-follower
# attends to, so the output contract is the last thing it reads.
REPAIR_NOTE = (
    "\n\nIMPORTANT — your previous reply could not be used: {why}. Respond again "
    "and follow the output contract EXACTLY. No explanations, no markdown fences, "
    "no repetition."
)

_JSON_CONTRACT = (
    'Output ONLY a single JSON object. No prose before or after it, no markdown '
    'code fences.'
)


def complete_json(client, *, system: str, user: str, cfg: Config,
                  attempts: int = 2, budget=None, **complete_kw) -> tuple:
    """Call the model demanding a parseable JSON object, repairing as needed.

    Returns (obj, raw_text, attempts_used): obj is the parsed dict (or None if the
    model never produced one), raw_text its last reply, attempts_used how many
    calls were spent. The caller decides what None means (fall back, skip, ask) —
    this function's only job is to make 'the model must emit JSON' a verified
    property of the run instead of an assumption. Extra kwargs (e.g. max_tokens)
    pass through to the client.
    """
    last_text = ""
    for i in range(1, max(1, attempts) + 1):
        prompt = user if i == 1 else user + REPAIR_NOTE.format(why=why)
        res = client.complete(system=system, user=prompt, cfg=cfg, **complete_kw)
        if budget is not None:
            budget.charge_model_call(res.input_tokens, res.output_tokens)
        last_text = res.text or ""
        deg = looks_degenerate(last_text)
        obj = extract_json(last_text)
        if obj is not None and deg is None:
            return obj, last_text, i
        why = deg or "it was not a parseable JSON object"
    return None, last_text, max(1, attempts)
