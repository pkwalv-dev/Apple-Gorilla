"""Local video generation — text (or an image) to a clip, on your own GPU.

Runs Wan 2.2 TI2V-5B under ComfyUI by default: Apache-2.0, text-to-video and
image-to-video in one checkpoint, no filter in the weights, and the largest open
video model that fits an 8GB card. The model is chosen by `cfg.comfy_video_workflow`,
which names a graph under `ag/workflows/` — swapping models is swapping that file.

Same contract as `images.generate`: if nothing can produce a video, this raises with
an actionable message. AG never reports a clip it did not make.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import comfy
from .config import Config, STATE_DIR, ensure_dirs

VIDEO_DIR = STATE_DIR / "video"


@dataclass
class VideoResult:
    path: str
    prompt: str
    width: int
    height: int
    frames: int
    fps: int
    seconds: float
    workflow: str


def available(cfg: Config) -> bool:
    return bool(getattr(cfg, "allow_video_gen", False))


def generate(prompt: str, cfg: Config, *, negative_prompt: str = "",
             seconds: Optional[float] = None, width: Optional[int] = None,
             height: Optional[int] = None, steps: Optional[int] = None,
             seed: Optional[int] = None) -> VideoResult:
    """Generate one clip. Raises RuntimeError with a usable message on any failure."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("empty video prompt")
    if not available(cfg):
        raise RuntimeError("video generation is disabled (allow_video_gen=false)")

    fps = int(getattr(cfg, "video_fps", 24) or 24)
    frames = int(getattr(cfg, "video_frames", 49) or 49)
    if seconds:
        # Wan's latent length wants 4n+1 frames; round to the nearest valid count.
        frames = max(5, int(round((float(seconds) * fps - 1) / 4)) * 4 + 1)
    w = int(width or getattr(cfg, "video_width", 704))
    h = int(height or getattr(cfg, "video_height", 400))

    name = getattr(cfg, "comfy_video_workflow", "wan22_ti2v_txt2vid")
    graph = comfy.load_workflow(name)          # raises if the template is missing
    if getattr(cfg, "comfy_autostart", True) and not comfy.reachable(cfg):
        comfy.ensure_running(cfg)
    filled = comfy.fill(graph, {
        comfy.PROMPT: prompt,
        comfy.NEGATIVE: negative_prompt or "",
        comfy.SEED: int(seed if seed is not None else random.randrange(2 ** 31)),
        comfy.STEPS: int(steps or getattr(cfg, "video_steps", 20)),
        comfy.WIDTH: w, comfy.HEIGHT: h,
        comfy.FRAMES: frames, comfy.FPS: fps,
    })
    # A clip is minutes of GPU time even on a small model; the wait scales with it.
    outputs = comfy.run(filled, cfg, timeout=max(900.0, frames * 30.0))

    ensure_dirs()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    fname, data = outputs[0]
    suffix = Path(fname).suffix or ".mp4"
    from .images import _slug
    path = VIDEO_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(prompt)}{suffix}"
    path.write_bytes(data)
    return VideoResult(path=str(path), prompt=prompt, width=w, height=h,
                       frames=frames, fps=fps, seconds=round(frames / fps, 2),
                       workflow=name)


def status(cfg: Config) -> dict:
    """What AG can say about video generation without generating anything."""
    info = comfy.status(cfg)
    return {"enabled": available(cfg), "backend": "comfy",
            "host": info["host"], "reachable": info["reachable"],
            "installed": info["installed"], "workflow": info.get("video_workflow", ""),
            "models": info.get("video_models", [])}
