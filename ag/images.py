"""Local image generation via an Automatic1111/Forge-compatible Stable Diffusion API.

Keyless and offline: AG POSTs to `{sd_host}/sdapi/v1/txt2img` on a Stable Diffusion
web server you run yourself (Automatic1111, Forge, reForge, ...). No third-party API,
no key, no per-image cost — the prompt never leaves your machine. Generated PNGs are
saved under state/images/ and also returned as a base64 data URL for the web app.

If no SD server is reachable the calls fail with a clear, actionable error rather than
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


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (s[:n] or "image")


def generate(prompt: str, cfg: Config, *, negative_prompt: str = "",
             steps: Optional[int] = None, width: Optional[int] = None,
             height: Optional[int] = None) -> ImageResult:
    """Generate one image from a text prompt via the local SD server.

    Raises RuntimeError with a helpful message if the server is unreachable or returns
    no image — callers surface this to the user instead of fabricating a result.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("empty image prompt")
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
