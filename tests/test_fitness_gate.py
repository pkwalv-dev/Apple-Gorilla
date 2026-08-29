"""Tests for the empirical fitness gate — the keep-if-better selection that turns
`evolve` from a blind tuner into measured self-improvement."""
import json
import types

import pytest

from ag import evolve as ev, archive, backup
from ag.config import Config
from ag.evolve import _fitness_verdict
from ag.model import ModelResult


# --- the adoption rule is a pure, directly-testable function ---------------

def test_verdict_improved():
    assert _fitness_verdict(5.0, 6.0, tol=0.05) == "improved"


def test_verdict_regressed():
    assert _fitness_verdict(6.0, 4.0, tol=0.05) == "regressed"


def test_verdict_neutral_within_tolerance():
    assert _fitness_verdict(6.0, 6.02, tol=0.05) == "neutral"
    assert _fitness_verdict(6.0, 5.97, tol=0.05) == "neutral"


def test_verdict_unknown_when_unmeasured():
    assert _fitness_verdict(None, 6.0, tol=0.05) == "unknown"
    assert _fitness_verdict(6.0, None, tol=0.05) == "unknown"


# --- end-to-end: a regression is rolled back, an improvement is adopted -----

class _PatchClient:
    """Fake model that always proposes one valid patch to an evolvable file."""

    def __init__(self, path, new_content):
        self._path, self._content = path, new_content

    def complete(self, *, system, user, cfg, max_tokens=None):
        return ModelResult(text=json.dumps(
            {"rationale": "test edit", "patches":
                [{"path": self._path, "new_content": self._content}]}))


def _harness(monkeypatch, tmp_path, fitnesses):
    """Wire a hermetic evolve: stub tests green, stub fitness, no git, temp archive."""
    monkeypatch.setattr(archive, "ARCHIVE_DIR", tmp_path / "arch")
    monkeypatch.setattr(archive, "LINEAGE_FILE", tmp_path / "arch" / "lineage.jsonl")
    monkeypatch.setattr(archive, "CACHE_FILE", tmp_path / "arch" / "cache.json")
    monkeypatch.setattr(ev, "run_tests", lambda: (True, "ok"))
    monkeypatch.setattr(backup, "git_commit_evolve", lambda *a, **k: None)
    seq = iter(fitnesses)
    monkeypatch.setattr(ev, "_measure_fitness", lambda *a, **k: types.SimpleNamespace(
        fitness=next(seq), pass_rate=0.5, per_task=[]))


def test_regression_is_rolled_back(monkeypatch, tmp_path):
    target = ev.ROOT / "profile/principles.md"
    original = target.read_text()
    _harness(monkeypatch, tmp_path, fitnesses=[6.0, 4.0])  # incumbent 6, candidate 4
    cfg = Config()
    cfg.autonomy_level = "guarded"
    cfg.fitness_gate = True
    client = _PatchClient("profile/principles.md", original + "\n<!-- edit -->\n")
    try:
        res = ev.evolve(client, cfg, dry_run=False)
        assert res.adopted is False
        assert res.rolled_back is True
        assert res.verdict == "regressed"
        assert res.fitness_delta == -2.0
        assert target.read_text() == original  # instant rollback restored the file
    finally:
        target.write_text(original)


def test_improvement_is_adopted(monkeypatch, tmp_path):
    target = ev.ROOT / "profile/principles.md"
    original = target.read_text()
    _harness(monkeypatch, tmp_path, fitnesses=[4.0, 6.0])  # incumbent 4, candidate 6
    cfg = Config()
    cfg.autonomy_level = "guarded"
    cfg.fitness_gate = True
    client = _PatchClient("profile/principles.md", original + "\n<!-- better -->\n")
    try:
        res = ev.evolve(client, cfg, dry_run=False)
        assert res.adopted is True
        assert res.rolled_back is False
        assert res.verdict == "improved"
        assert res.fitness_delta == 2.0
        # The improvement is recorded in the archive with its measured delta.
        hist = archive.adopted_history()
        assert hist and hist[0]["delta"] == 2.0
    finally:
        target.write_text(original)


def test_neutral_change_is_adopted(monkeypatch, tmp_path):
    target = ev.ROOT / "profile/principles.md"
    original = target.read_text()
    _harness(monkeypatch, tmp_path, fitnesses=[6.0, 6.0])  # unchanged score
    cfg = Config()
    cfg.autonomy_level = "guarded"
    cfg.fitness_gate = True
    client = _PatchClient("profile/principles.md", original + "\n<!-- lateral -->\n")
    try:
        res = ev.evolve(client, cfg, dry_run=False)
        assert res.adopted is True
        assert res.verdict == "neutral"
    finally:
        target.write_text(original)


def test_measure_fitness_subprocess_roundtrip():
    # The REAL _measure_fitness shells out to `python -m ag bench --json` so it scores
    # freshly-written source, not stale in-memory modules. Guards the JSON contract:
    # if `--json` ever stops emitting clean JSON, this breaks. Dry backend -> no network.
    from ag.model import DryRunClient
    r = ev._measure_fitness(DryRunClient(), Config())
    assert hasattr(r, "fitness") and isinstance(r.fitness, float)
    assert len(r.per_task) >= 10  # the seed suite ran in the subprocess


def test_fitness_gate_off_skips_measurement(monkeypatch, tmp_path):
    # With the gate disabled, evolve must not call the (costly) benchmark at all.
    target = ev.ROOT / "profile/principles.md"
    original = target.read_text()
    monkeypatch.setattr(archive, "ARCHIVE_DIR", tmp_path / "arch")
    monkeypatch.setattr(archive, "LINEAGE_FILE", tmp_path / "arch" / "lineage.jsonl")
    monkeypatch.setattr(archive, "CACHE_FILE", tmp_path / "arch" / "cache.json")
    monkeypatch.setattr(ev, "run_tests", lambda: (True, "ok"))
    monkeypatch.setattr(backup, "git_commit_evolve", lambda *a, **k: None)

    def boom(*a, **k):
        raise AssertionError("benchmark must not run when fitness_gate is off")
    monkeypatch.setattr(ev, "_measure_fitness", boom)

    cfg = Config()
    cfg.autonomy_level = "guarded"
    cfg.fitness_gate = False
    client = _PatchClient("profile/principles.md", original + "\n<!-- x -->\n")
    try:
        res = ev.evolve(client, cfg, dry_run=False)
        assert res.adopted is True           # adopts on the safety gate alone
        assert res.incumbent_fitness is None  # never measured
    finally:
        target.write_text(original)
