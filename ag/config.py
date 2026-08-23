"""Configuration and filesystem paths for Apple-Gorilla."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

# Repo root = parent of the `ag/` package directory.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
PROFILE_DIR = ROOT / "profile"
STATE_DIR = ROOT / "state"
RUNS_DIR = STATE_DIR / "runs"
VERSIONS_DIR = STATE_DIR / "versions"


@dataclass
class Config:
    """Runtime configuration. Persisted to config.json (itself an evolvable file)."""

    backend: str = "auto"                     # auto | anthropic | ollama | dry
    model: str = "claude-opus-5"
    ollama_model: str = "llama3.1"            # used when backend == ollama
    ollama_host: str = "http://localhost:11434"
    ollama_keep_alive: str = "30m"            # keep the model resident between calls
    ollama_options: dict = field(default_factory=dict)  # e.g. {"num_ctx": 8192}
    effort: str = "high"                      # low | medium | high | xhigh | max
    max_output_tokens: int = 32000            # main answer generation
    meta_output_tokens: int = 16000           # optimizer / critic / evolve calls
    max_iterations: int = 2                   # critique->revise rounds
    critic_pass_threshold: float = 8.0        # 0..10; at/above this we stop iterating
    speed_budget_s: float = 30.0              # target wall-clock for a full speed score
    score_weights: dict = field(default_factory=lambda: {  # directed-evolution axes
        "accuracy": 0.5, "quality": 0.3, "speed": 0.2,
    })
    autonomy_level: str = "guarded"           # manual | guarded | never
    allow_external_tools: bool = False        # default-deny for Chrome/network/etc.
    allow_web: bool = True                     # AG's standing internet access
    max_snapshots: int = 20                    # cap on kept source snapshots
    max_runs: int = 100                        # cap on kept run telemetry logs
    evolve_branch: str = "ag/evolve"           # AG's self-commits land here, never main
    evolvable_paths: List[str] = field(default_factory=lambda: [
        "ag/prompts.py",
        "ag/tools/web.py",       # internet code self-improves (highest-churn area)
        "profile/principles.md",
        "config.json",
    ])

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        if path.exists():
            data = json.loads(path.read_text())
            known = {k: data[k] for k in data if k in cls.__dataclass_fields__}
            return cls(**known)
        return cls()

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def ensure_dirs() -> None:
    for d in (STATE_DIR, RUNS_DIR, VERSIONS_DIR, PROFILE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for keep in (RUNS_DIR / ".gitkeep", VERSIONS_DIR / ".gitkeep"):
        if not keep.exists():
            keep.write_text("")
