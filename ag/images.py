"""Local image generation — keyless, offline, on your own GPU.

Two backends, chosen by `cfg.image_backend` (default "auto"):

- **comfy**  — ComfyUI, running Chroma1-HD by default: 8.9B, Apache-2.0, a de-distilled
  FLUX.1-schnell retrained with no safety filter, and unlike FLUX it honours real CFG
  and negative prompts. The model is named by a workflow file, not by code (see
  `ag/comfy.py`), so changing models is changing a JSON file.
- **a1111** — an Automatic1111/Forge-compatible server: AG POSTs to
  `{sd_host}/sdapi/v1/txt2img`. SD-family checkpoints only.

"auto" prefers ComfyUI when it answers and falls back to A1111, so an existing Stable
Diffusion install keeps working untouched. Either way no third-party API is involved,
there is no key and no per-image cost, and the prompt never leaves your machine.
Generated PNGs are saved under state/images/ and returned as a base64 data URL for the
web app.

If nothing can render, the call fails with a clear, actionable error rather than
hanging — AG never pretends an image was made.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config, STATE_DIR, ensure_dirs

IMAGES_DIR = STATE_DIR / "images"


@dataclass
class ImageResult:
    path: str            # absolute path of the saved PNG
    data_url: str        # "data:image/png;base64,..." for direct display
    prompt: str
    width: int
    height: int
    steps: int


def sd_reachable(cfg: Config, timeout: float = 1.5) -> bool:
    """True if a Stable Diffusion web API answers at cfg.sd_host."""
    url = cfg.sd_host.rstrip("/") + "/sdapi/v1/sd-models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= getattr(r, "status", 200) < 500
    except Exception:
        return False


def _is_local_host(host: str) -> bool:
    from .model import _is_local_host as _lh
    return _lh(host)


def _find_sd_launcher():
    """Locate an installed Automatic1111/Forge launcher. Returns (argv, cwd) or None.

    We only *launch* an existing install; we never install SD or download models.
    """
    import os
    home = Path.home()
    dir_names = ["stable-diffusion-webui", "stable-diffusion-webui-forge",
                 "stable-diffusion-webui-directml", "forge", "sd-forge", "sdnext",
                 "automatic", "SD"]
    roots = [home, Path.cwd()]
    for env in ("USERPROFILE", "LOCALAPPDATA", "ProgramFiles", "ProgramW6432"):
        v = os.environ.get(env)
        if v:
            roots.append(Path(v))
    win_scripts = ["webui-user.bat", "webui.bat"]
    nix_scripts = ["webui.sh"]
    scripts = win_scripts if os.name == "nt" else nix_scripts
    seen = set()
    for root in roots:
        for d in dir_names:
            base = root / d
            if base in seen:
                continue
            seen.add(base)
            for s in scripts:
                p = base / s
                try:
                    if p.exists():
                        if os.name == "nt":
                            return (["cmd", "/c", str(p)], str(base))
                        return (["bash", str(p), "--api"], str(base))
                except OSError:
                    continue
    return None


def _spawn_detached(argv, cwd, extra_env=None):
    import os
    import subprocess
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    try:
        if os.name == "nt":
            flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             creationflags=flags, close_fds=True)
        else:
            subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        return True
    except Exception:
        return False


def _launch_sd(cfg: Config) -> bool:
    """Best-effort launch of an installed SD server with its API enabled."""
    import os
    import shlex
    cmd = (getattr(cfg, "sd_cmd", "") or "").strip()
    if cmd:
        # Treat a bare path to a script/dir specially; otherwise run as a command.
        p = Path(cmd)
        if p.exists() and p.is_file():
            if os.name == "nt":
                return _spawn_detached(["cmd", "/c", str(p)], str(p.parent),
                                       {"COMMANDLINE_ARGS": "--api"})
            return _spawn_detached(["bash", str(p), "--api"], str(p.parent))
        argv = cmd.split() if os.name == "nt" else shlex.split(cmd)
        if "--api" not in argv:
            argv.append("--api")
        return _spawn_detached(argv, str(Path.cwd()))
    found = _find_sd_launcher()
    if not found:
        return False
    argv, cwd = found
    # webui-user.bat reads COMMANDLINE_ARGS; ensure the API is on for .bat launches.
    extra = {"COMMANDLINE_ARGS": "--api"} if argv and str(argv[-1]).lower().endswith(".bat") else None
    return _spawn_detached(argv, cwd, extra)


def ensure_sd_running(cfg: Config, *, timeout: float = 180.0) -> bool:
    """Make sure a local SD server is up, launching an installed one if needed.

    Best-effort and never raises. Returns True once the API is reachable. Only a local
    host is auto-started, gated by cfg.sd_autostart. Startup includes loading a model,
    so the first call can take a while (hence the generous timeout).
    """
    import time as _t
    if sd_reachable(cfg):
        return True
    if not getattr(cfg, "sd_autostart", True) or not _is_local_host(cfg.sd_host):
        return False
    if not _launch_sd(cfg):
        return False
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        if sd_reachable(cfg, timeout=2.0):
            return True
        _t.sleep(2.0)
    return False


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (s[:n] or "image")


def backend_for(cfg: Config) -> str:
    """Which media backend this run uses: "comfy" or "a1111".

    "auto" prefers ComfyUI when it answers, because that is where the unfiltered
    open-weight models run — and falls back to A1111 so an existing Stable Diffusion
    install keeps working exactly as before. An explicit setting is obeyed as given.
    """
    choice = (getattr(cfg, "image_backend", "auto") or "auto").lower()
    if choice in ("comfy", "a1111"):
        return choice
    from . import comfy
    if comfy.reachable(cfg):
        return "comfy"
    if sd_reachable(cfg):
        return "a1111"
    # Neither is up: prefer the one that is at least installed, else ComfyUI, whose
    # error message names the model AG expects.
    return "a1111" if _find_sd_launcher() else "comfy"


def generate(prompt: str, cfg: Config, *, negative_prompt: str = "",
             steps: Optional[int] = None, width: Optional[int] = None,
             height: Optional[int] = None,
             seed: Optional[int] = None) -> ImageResult:
    """Generate one image, via whichever local backend this machine has.

    Raises RuntimeError with a helpful message if nothing can render it — callers
    surface that instead of fabricating a result.
    """
    if backend_for(cfg) == "comfy":
        return _generate_comfy(prompt, cfg, negative_prompt=negative_prompt,
                               steps=steps, width=width, height=height, seed=seed)
    return _generate_a1111(prompt, cfg, negative_prompt=negative_prompt,
                           steps=steps, width=width, height=height)


def _generate_comfy(prompt: str, cfg: Config, *, negative_prompt: str = "",
                    steps: Optional[int] = None, width: Optional[int] = None,
                    height: Optional[int] = None,
                    seed: Optional[int] = None) -> ImageResult:
    """Chroma1-HD (or whatever `comfy_image_workflow` names) through ComfyUI."""
    import random
    from . import comfy
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("empty image prompt")
    name = getattr(cfg, "comfy_image_workflow", "chroma1hd_txt2img")
    graph = comfy.load_workflow(name)
    if getattr(cfg, "comfy_autostart", True) and not comfy.reachable(cfg):
        comfy.ensure_running(cfg)
    w = int(width or cfg.sd_width)
    h = int(height or cfg.sd_height)
    st = int(steps or cfg.sd_steps)
    filled = comfy.fill(graph, {
        comfy.PROMPT: prompt, comfy.NEGATIVE: negative_prompt or "",
        comfy.SEED: int(seed if seed is not None else random.randrange(2 ** 31)),
        comfy.STEPS: st, comfy.WIDTH: w, comfy.HEIGHT: h,
    })
    outputs = comfy.run(filled, cfg)
    fname, raw = outputs[0]
    ensure_dirs()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGES_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(prompt)}.png"
    path.write_bytes(raw)
    return ImageResult(path=str(path),
                       data_url="data:image/png;base64," +
                                base64.b64encode(raw).decode("ascii"),
                       prompt=prompt, width=w, height=h, steps=st)


def _generate_a1111(prompt: str, cfg: Config, *, negative_prompt: str = "",
                    steps: Optional[int] = None, width: Optional[int] = None,
                    height: Optional[int] = None) -> ImageResult:
    """Generate one image from a text prompt via the local SD server.

    Raises RuntimeError with a helpful message if the server is unreachable or returns
    no image — callers surface this to the user instead of fabricating a result.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("empty image prompt")
    # Auto-launch an installed local SD server if it isn't already up.
    if getattr(cfg, "sd_autostart", True) and not sd_reachable(cfg):
        ensure_sd_running(cfg)
    host = cfg.sd_host.rstrip("/")
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "steps": int(steps or cfg.sd_steps),
        "width": int(width or cfg.sd_width),
        "height": int(height or cfg.sd_height),
        "sampler_name": getattr(cfg, "sd_sampler", "Euler a"),
    }
    req = urllib.request.Request(
        host + "/sdapi/v1/txt2img",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach a Stable Diffusion server at {host} ({e}). Start one "
            "(Automatic1111/Forge) with its API enabled (--api), or set cfg.sd_host."
        ) from e

    images = data.get("images") or []
    if not images:
        raise RuntimeError("the image server returned no image "
                           "(check the model is loaded and the prompt is allowed)")
    b64 = images[0].split(",", 1)[-1]        # tolerate a leading data: prefix
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise RuntimeError(f"could not decode the returned image: {e}") from e

    ensure_dirs()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(prompt)}.png"
    path = IMAGES_DIR / fname
    path.write_bytes(raw)
    return ImageResult(
        path=str(path),
        data_url="data:image/png;base64," + b64,
        prompt=prompt,
        width=payload["width"],
        height=payload["height"],
        steps=payload["steps"],
    )
