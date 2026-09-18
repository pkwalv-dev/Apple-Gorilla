"""Ensure the repo root is importable as the `ag` package during tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Never collect runtime state as tests. Acquired skills live under state/skills/ and
# each ships its own test_skill.py (gated in isolation at acquire time); collecting
# them here would clash on basename and — critically — break the evolve test gate,
# which runs `pytest` at the repo root. Keep AG's own suite the only thing collected.
collect_ignore_glob = ["state/*", "state/**/*"]
