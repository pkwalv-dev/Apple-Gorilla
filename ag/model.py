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
from typing import Optional

from .config import Config


@dataclass
class ModelResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stopped: str = "end_turn"
    dry_run: bool = False


class DryRunClient:
    """Deterministic offline stand-in for the API.

    It echoes structured, plausible output so the pipeline (optimize/execute/
    critique/iterate/evolve) can be exercised and tested with no network or key.
    """

    def complete(self, *, system: str, user: str, cfg: Config,
                 max_tokens: Optional[int] = None) -> ModelResult:
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
        if not _has_anthropic_creds():
            # No API key: authenticate with the `ant auth login` OAuth profile. The
            # SDK does NOT auto-read that credentials file, so we load the bearer
            # token and pass it explicitly, plus the OAuth beta header /v1/messages
            # requires. (A bare client would have no auth at all and fail obscurely.)
            token = _load_oauth_token()
            if token is None:
                raise RuntimeError(
                    "No ANTHROPIC_API_KEY and no readable OAuth profile. Set a key, "
                    "or run `ant auth login` (or sign in via Claude Code)."
                )
            headers["anthropic-beta"] = "oauth-2025-04-20"
            kwargs["auth_token"] = token
        self._client = anthropic.Anthropic(**kwargs)

    def complete(self, *, system: str, user: str, cfg: Config,
                 max_tokens: Optional[int] = None) -> ModelResult:
        mt = max_tokens or cfg.max_output_tokens
        # Stream for large generations to avoid HTTP timeouts (skill guidance).
        with self._client.messages.stream(
            model=cfg.model,
            max_tokens=mt,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": cfg.effort},
            messages=[{"role": "user", "content": user}],
        ) as stream:
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
                 max_tokens: Optional[int] = None) -> ModelResult:
        import urllib.error
        import urllib.request

        # Merge tuned options (num_ctx, num_gpu, temperature, ...) from config; the
        # answer-length cap always wins. keep_alive keeps the model resident across
        # the pipeline's 3-4 calls per run, avoiding costly reloads (big latency win).
        options = dict(getattr(cfg, "ollama_options", {}) or {})
        options["num_predict"] = max_tokens or cfg.max_output_tokens
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": getattr(cfg, "ollama_keep_alive", "30m"),
            "options": options,
        }
        req = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._host} ({e}). "
                f"Is `ollama serve` running and `{self._model}` pulled?"
            ) from e
        text = (data.get("message") or {}).get("content", "")
        return ModelResult(
            text=text,
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            model=self._model,
        )


def _has_anthropic_creds() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


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
        return OllamaClient(cfg)
    if choice == "anthropic":
        return ApiClient()
    # auto: use Claude when an API key OR an OAuth profile is present.
    if _has_anthropic_creds() or has_oauth_profile():
        try:
            return ApiClient()
        except Exception:
            return DryRunClient()
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
