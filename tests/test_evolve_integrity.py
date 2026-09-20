"""Verifier integrity: evolution must not be able to weaken its own gates.

The attack chain that matters: config.json is evolvable and `evolvable_paths`
lives IN it, so without explicit guards a candidate could

  1. add `ag/evolve.py` (or `ag/bench.py`, the fitness ruler) to evolvable_paths,
     then edit the gates in a LATER cycle — Gate 1 can't catch it, the tests all
     still pass while the fitness gate is being hollowed out;
  2. soften the safety knobs directly (`fitness_gate: false`, `fitness_k: 0`);
  3. change the measuring instrument the bench subprocess reads (fewer tasks, a
     different model, an easier mode) — re-grading itself on a friendlier ruler.

These are the failures that turn a self-improving system into an unverifiable
one, so the guards live in code, not config.
"""
from __future__ import annotations

import json

from ag import evolve


# --- protected files ----------------------------------------------------------

def test_gate_files_are_never_valid_patch_targets():
    for gate_file in ("ag/evolve.py", "ag/bench.py", "ag/benchgen.py",
                      "ag/permissions.py", "ag/scoring.py", "conftest.py"):
        err = evolve._validate_patch(gate_file, "# pass\n")
        assert err and "gate" in err.lower() or "verification" in err.lower(), gate_file


def test_a_normal_evolvable_file_still_validates():
    assert evolve._validate_patch("ag/prompts.py",
                                  'EXECUTOR_SYSTEM = "x"\n') is None
    # and syntax validation still works for it
    err = evolve._validate_patch("ag/prompts.py", "def broken(:\n")
    assert err and "syntax" in err


def test_config_patch_cannot_extend_evolvable_paths_to_the_gates():
    incumbent = json.loads((evolve.ROOT / "config.json").read_text())
    attack = dict(incumbent)
    attack["evolvable_paths"] = incumbent["evolvable_paths"] + ["ag/evolve.py"]
    err = evolve._validate_config_patch(json.dumps(attack))
    assert err and "evolvable_paths" in err


def test_config_patch_cannot_soften_the_fitness_gate():
    incumbent = json.loads((evolve.ROOT / "config.json").read_text())
    for key, weakened in (("fitness_gate", False), ("bench_validate", False),
                          ("fitness_k", 0.0), ("bench_samples", 1),
                          ("bench_generated_n", 1), ("autonomy_level", "never")):
        attack = dict(incumbent)
        attack[key] = weakened
        err = evolve._validate_config_patch(json.dumps(attack))
        assert err and key in err, key


def test_config_patch_cannot_swap_the_measured_model():
    """The bench subprocess reloads config.json, so swapping ollama_model in a
    candidate would re-grade the candidate on a different brain than the incumbent
    was measured with — a silently rigged comparison."""
    incumbent = json.loads((evolve.ROOT / "config.json").read_text())
    attack = dict(incumbent)
    attack["ollama_model"] = "much-bigger-model:70b"
    err = evolve._validate_config_patch(json.dumps(attack))
    assert err and "ollama_model" in err


def test_safe_config_tuning_is_still_allowed():
    """The guard must not freeze all of config.json — speed budgets, themes and
    prompt behaviour are exactly what self-tuning is FOR."""
    incumbent = json.loads((evolve.ROOT / "config.json").read_text())
    safe = dict(incumbent)
    safe["speed_budget_s"] = 12.0
    safe["budget_model_calls"] = 20
    assert evolve._validate_config_patch(json.dumps(safe)) is None


def test_an_identical_config_is_allowed():
    incumbent = (evolve.ROOT / "config.json").read_text()
    assert evolve._validate_config_patch(incumbent) is None


# --- the hash pin ---------------------------------------------------------------

def test_gate_hashes_are_stable_and_cover_real_files():
    hashes = evolve._gate_hashes()
    assert "ag/evolve.py" in hashes and "ag/bench.py" in hashes
    again = evolve._gate_hashes()
    assert hashes == again
    for digest in hashes.values():
        assert len(digest) == 64


def test_hash_pin_detects_a_tampered_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve, "ROOT", tmp_path)
    gate = tmp_path / "ag" / "bench.py"
    gate.parent.mkdir(parents=True)
    gate.write_text("# honest ruler\n")
    hashes = evolve._gate_hashes()
    gate.write_text("# fitness = 10.0  # (: always\n")
    assert evolve._gate_hashes() != hashes


# --- the regression corpus ------------------------------------------------------

def test_regressions_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve, "REGRESSIONS_FILE",
                        tmp_path / "evolve" / "regressions.jsonl")
    assert evolve.regressions() == []
    evolve._record_regression("tests", reason="tests failed: 1 error",
                              changed=["ag/theme.py"], rationale="darken accent",
                              incumbent=7.5)
    evolve._record_regression("overfit", reason="held-out dropped 8->5",
                              incumbent=8.0, candidate=5.0)
    recs = evolve.regressions()
    assert [r["gate"] for r in recs] == ["tests", "overfit"]
    assert recs[1]["candidate_fitness"] == 5.0
    assert "when" in recs[0]


def test_recent_regressions_reach_the_proposer_briefing(tmp_path, monkeypatch):
    """The loop learns from its rejections: the corpus is fed back into the
    briefing so a vetoed direction is not re-proposed forever."""
    monkeypatch.setattr(evolve, "REGRESSIONS_FILE",
                        tmp_path / "evolve" / "regressions.jsonl")
    evolve._record_regression("integrity", reason="verifier files changed")
    from ag.config import Config
    briefing = evolve._direction(Config(), telemetry=[])
    assert "recent_rejected_proposals" in briefing
    assert briefing["recent_rejected_proposals"][0]["gate"] == "integrity"
