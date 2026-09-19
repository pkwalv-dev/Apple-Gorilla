"""ComfyUI as AG's media backend — unfiltered image and video generation, locally.

Why this exists alongside `images.py`'s Automatic1111 path: A1111 runs SD-family
checkpoints. The open-weight models that carry no safety filter *in the weights* —
Chroma1-HD (images) and Wan 2.2 TI2V-5B (video) — are not SD-family and run under
ComfyUI. So AG speaks ComfyUI's API as well, and prefers it when it answers.

The API is three calls:

    POST /prompt            {"prompt": <graph>, "client_id": ...}  -> {"prompt_id"}
    GET  /history/<id>                                             -> outputs per node
    GET  /view?filename=&subfolder=&type=                          -> the bytes

**Workflows are data.** AG does not build graphs in Python — it loads a graph you
exported from ComfyUI itself ("Workflow → Export (API)") and substitutes values into
it by placeholder. That matters because node names and socket names change between
ComfyUI releases and between models: a graph AG hardcodes is a graph that rots. A new
model is a new JSON file under `ag/workflows/` (or `state/workflows/`, which wins),
not a patch to this module.

Nothing here downloads a model or installs ComfyUI. AG will *start* an install it
finds, and otherwise says plainly what is missing.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, STATE_DIR

WORKFLOW_DIR = Path(__file__).resolve().parent / "workflows"
USER_WORKFLOW_DIR = STATE_DIR / "workflows"

# Placeholders a workflow template may contain. Anything not supplied is left alone,
# so a graph can opt out of any of them.
PROMPT = "__AG_PROMPT__"
NEGATIVE = "__AG_NEGATIVE__"
SEED = "__AG_SEED__"
STEPS = "__AG_STEPS__"
WIDTH = "__AG_WIDTH__"
HEIGHT = "__AG_HEIGHT__"
FRAMES = "__AG_FRAMES__"
FPS = "__AG_FPS__"


# --- reachability and startup ----------------------------------------------
def reachable(cfg: Config, timeout: float = 1.5) -> bool:
    """True if a ComfyUI server answers at cfg.comfy_host."""
    url = cfg.comfy_host.rstrip("/") + "/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= getattr(r, "status", 200) < 500
    except Exception:
        return False


def _interpreter_for(comfy_dir: Path, fallback: str) -> str:
    """The Python that can actually run this ComfyUI.

    ComfyUI needs a CUDA torch build, which AG's own interpreter has no reason to
    have — so a venv beside main.py, or the portable build's bundled runtime, is
    preferred over whatever is running AG. Launching with the wrong interpreter fails
    at `import torch`, several seconds into a detached process where nobody sees it.
    """
    names = ("python.exe", "python")
    for rel in (("python_embeded",), (".venv", "Scripts"), (".venv", "bin"),
                ("venv", "Scripts"), ("venv", "bin")):
        base = comfy_dir.joinpath(*rel)
        if not base.exists():
            base = comfy_dir.parent.joinpath(*rel)     # the portable build's sibling
        for n in names:
            p = base / n
            try:
                if p.exists():
                    return str(p)
            except OSError:
                continue
    return fallback


def _find_comfy() -> Optional[Tuple[list, str]]:
    """Locate an installed ComfyUI. Returns (argv, cwd) or None. Never installs."""
    import os
    import sys
    home = Path.home()
    roots = [home, Path.cwd(), home / "Documents", home / "Documents" / "GitHub"]
    for env in ("USERPROFILE", "LOCALAPPDATA", "ProgramFiles", "ProgramW6432"):
        v = os.environ.get(env)
        if v:
            roots.append(Path(v))
    names = ["ComfyUI", "comfyui", "ComfyUI_windows_portable"]
    for root in roots:
        for n in names:
            base = root / n
            # The portable build nests the real tree one level down.
            for cand in (base, base / "ComfyUI"):
                try:
                    if (cand / "main.py").exists():
                        return ([_interpreter_for(cand, sys.executable), "main.py"],
                                str(cand))
                except OSError:
                    continue
    return None


def ensure_running(cfg: Config, *, timeout: float = 240.0) -> bool:
    """Start an installed ComfyUI if it isn't up. Best-effort; never raises.

    Only a local host is auto-started. First start loads a multi-GB checkpoint, hence
    the generous wait.
    """
    if reachable(cfg):
        return True
    if not getattr(cfg, "comfy_autostart", True):
        return False
    from .images import _is_local_host, _spawn_detached
    if not _is_local_host(cfg.comfy_host):
        return False
    cmd = (getattr(cfg, "comfy_cmd", "") or "").strip()
    if cmd:
        import shlex
        import os
        argv = cmd.split() if os.name == "nt" else shlex.split(cmd)
        started = _spawn_detached(argv, str(Path.cwd()))
    else:
        found = _find_comfy()
        if not found:
            return False
        argv, cwd = found
        started = _spawn_detached(argv, cwd)
    if not started:
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if reachable(cfg, timeout=2.0):
            return True
        time.sleep(2.0)
    return False


# --- workflow templates -----------------------------------------------------
def workflow_path(name: str) -> Optional[Path]:
    """Resolve a workflow by name: the user's copy wins over the shipped one."""
    stem = re.sub(r"[^A-Za-z0-9_.-]", "", name or "").removesuffix(".json")
    if not stem:
        return None
    for d in (USER_WORKFLOW_DIR, WORKFLOW_DIR):
        p = d / f"{stem}.json"
        if p.exists():
            return p
    return None


def load_workflow(name: str) -> Dict[str, Any]:
    p = workflow_path(name)
    if p is None:
        raise RuntimeError(
            f"no ComfyUI workflow named {name!r} (looked in {USER_WORKFLOW_DIR} and "
            f"{WORKFLOW_DIR}). Export one from ComfyUI with Workflow → Export (API) "
            f"and save it there.")
    try:
        graph = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"workflow {p.name} is not valid JSON: {e}") from e
    if not isinstance(graph, dict) or not graph:
        raise RuntimeError(f"workflow {p.name} is not a ComfyUI API graph "
                           "(export with 'Export (API)', not the editor format)")
    return graph


def fill(graph: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    """Substitute placeholders throughout a graph, returning a new graph.

    A placeholder alone in a string becomes the typed value (so `__AG_STEPS__` lands
    as an int, which ComfyUI's validators require); a placeholder inside a longer
    string is interpolated as text, which is how a template adds a fixed style prefix
    to the prompt.
    """
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            if node in values:
                return values[node]
            for ph, val in values.items():
                if ph in node:
                    node = node.replace(ph, str(val))
            return node
        return node
    return walk(graph)


def meta(graph: Dict[str, Any]) -> Dict[str, Any]:
    """A template's `_ag` block: title, model, and where its weights come from.

    Keys starting with "_" are AG's own annotations, not nodes; they are stripped
    before the graph is queued.
    """
    m = graph.get("_ag")
    return m if isinstance(m, dict) else {}


def nodes(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Just the nodes — what ComfyUI is actually sent."""
    return {k: v for k, v in graph.items() if not k.startswith("_")}


def describe(graph: Dict[str, Any]) -> List[str]:
    """The model files a graph asks for — what must be present for it to run."""
    out: List[str] = []
    for node in nodes(graph).values():
        if not isinstance(node, dict):
            continue
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, str) and v.endswith((".safetensors", ".gguf", ".sft",
                                                  ".ckpt", ".pt")):
                if v not in out:
                    out.append(v)
    return out


# --- running a graph --------------------------------------------------------
def _post(cfg: Config, path: str, payload: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        cfg.comfy_host.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def _get(cfg: Config, path: str, timeout: float = 30.0) -> bytes:
    with urllib.request.urlopen(cfg.comfy_host.rstrip("/") + path,
                                timeout=timeout) as r:
        return r.read()


def run(graph: Dict[str, Any], cfg: Config, *,
        timeout: float = 900.0) -> List[Tuple[str, bytes]]:
    """Queue a graph and return its outputs as (filename, bytes), in node order.

    Raises RuntimeError with the server's own message on a rejected graph — a missing
    checkpoint or a node from an uninstalled extension is the common case, and the
    caller shows it verbatim rather than guessing.
    """
    cid = uuid.uuid4().hex
    try:
        res = _post(cfg, "/prompt", {"prompt": nodes(graph), "client_id": cid})
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:600]
        except Exception:
            pass
        raise RuntimeError(f"ComfyUI rejected the workflow ({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach ComfyUI at {cfg.comfy_host} ({e}). Start it, or set "
            "comfy_host / comfy_cmd in config.json.") from e

    pid = str(res.get("prompt_id") or "")
    if not pid:
        raise RuntimeError(f"ComfyUI accepted nothing back: {res!r}")

    deadline = time.time() + timeout
    entry: dict = {}
    while time.time() < deadline:
        try:
            hist = json.loads(_get(cfg, f"/history/{pid}").decode("utf-8") or "{}")
        except Exception:
            hist = {}
        entry = hist.get(pid) or {}
        status = (entry.get("status") or {})
        if entry.get("outputs"):
            break
        if status.get("status_str") == "error" or status.get("completed") is False \
                and status.get("status_str") in ("error", "failed"):
            raise RuntimeError(f"ComfyUI run failed: "
                               f"{json.dumps(status)[:600]}")
        time.sleep(1.0)
    else:
        raise RuntimeError(f"ComfyUI did not finish within {int(timeout)}s")

    out: List[Tuple[str, bytes]] = []
    for node_out in (entry.get("outputs") or {}).values():
        for key in ("images", "gifs", "videos", "audio"):
            for item in (node_out.get(key) or []):
                fn = str(item.get("filename") or "")
                if not fn:
                    continue
                q = urllib.parse.urlencode({
                    "filename": fn, "subfolder": str(item.get("subfolder") or ""),
                    "type": str(item.get("type") or "output")})
                try:
                    out.append((fn, _get(cfg, f"/view?{q}", timeout=120.0)))
                except Exception:
                    continue
    if not out:
        raise RuntimeError("ComfyUI produced no output file — check the workflow ends "
                           "in a Save node")
    return out


def validate(graph: Dict[str, Any], cfg: Config) -> List[str]:
    """Check a graph against the running server, returning problems in plain words.

    These templates are written from each model's documented node set, and ComfyUI
    renames sockets between releases — so rather than asking anyone to trust them, AG
    can check them against the server that will actually run them. Every node class,
    every required input, and every model filename is verified here before a single
    image is attempted. An empty list means the workflow will load.
    """
    problems: List[str] = []
    try:
        info = json.loads(_get(cfg, "/object_info", timeout=60.0).decode("utf-8"))
    except Exception as e:
        return [f"cannot read ComfyUI's node list at {cfg.comfy_host}: {e}"]

    for nid, node in nodes(graph).items():
        cls = str((node or {}).get("class_type") or "")
        spec = info.get(cls)
        if spec is None:
            problems.append(f"node {nid}: ComfyUI has no node class {cls!r} "
                            "(a custom node may need installing)")
            continue
        given = set((node.get("inputs") or {}).keys())
        required = (spec.get("input") or {}).get("required") or {}
        known = set(required) | set((spec.get("input") or {}).get("optional") or {})
        for miss in sorted(set(required) - given):
            problems.append(f"node {nid} ({cls}): missing required input {miss!r}")
        for extra in sorted(given - known):
            problems.append(f"node {nid} ({cls}): unknown input {extra!r}")
        # A loader names a file the server must actually have on disk. An EMPTY
        # choice list is the important case, not a skippable one: it means that
        # models directory is bare, which is exactly what a fresh install looks like.
        for key, val in (node.get("inputs") or {}).items():
            choices = (required.get(key) or [None])[0]
            if isinstance(val, str) and isinstance(choices, list) \
                    and val not in choices:
                where = ("that directory is empty" if not choices else
                         "server has " + ", ".join(map(str, choices[:3])))
                problems.append(f"node {nid} ({cls}): {key}={val!r} is not "
                                f"installed ({where})")
    return problems


def status(cfg: Config) -> dict:
    """What AG can tell the user about the media backend without generating anything."""
    up = reachable(cfg)
    info: dict = {"host": cfg.comfy_host, "reachable": up,
                  "installed": bool(_find_comfy()) or bool(getattr(cfg, "comfy_cmd", ""))}
    for kind, wf in (("image", getattr(cfg, "comfy_image_workflow", "")),
                     ("video", getattr(cfg, "comfy_video_workflow", ""))):
        p = workflow_path(wf)
        info[f"{kind}_workflow"] = wf
        info[f"{kind}_models"] = describe(load_workflow(wf)) if p else []
    return info
