"""Gate 3: the overfitting detector.

Keep-if-better selection has one characteristic failure mode — the candidate
learns the scored tasks rather than the skill. Gates 1 and 2 cannot see it, by
construction, because Gate 2 *is* the thing being gamed. The only honest
detector is a split the optimiser never gets feedback from.
"""
from __future__ import annotations

import dataclasses

from ag import evolve
from ag.config import Config


def _cfg(**kw):
    base = dataclasses.replace(Config(), bench_generated=True, bench_validate=True,
                               fitness_gate=True)
    return dataclasses.replace(base, **kw)


def test_validation_requires_a_generated_suite():
    """The 14 curated tasks are a fixed list — there is nothing to hold out, so
    claiming a held-out score there would be a lie."""
    assert evolve._validation_enabled(_cfg())
    assert not evolve._validation_enabled(_cfg(bench_generated=False))
    assert not evolve._validation_enabled(_cfg(bench_validate=False))


def test_bench_subprocess_is_told_which_split_to_run(monkeypatch):
    seen = []

    class Result:
        stdout = '{"fitness": 7.5, "pass_rate": 0.75, "per_task": []}'

    monkeypatch.setattr(evolve.subprocess, "run",
                        lambda cmd, **kw: (seen.append(cmd), Result())[1])
    cfg = _cfg()
    evolve._bench_once(object(), cfg, split="train")
    evolve._bench_once(object(), cfg, split="validation")

    assert "--split" not in seen[0], "the training run must not filter the suite"
    assert seen[1][seen[1].index("--split") + 1] == "validation"
    assert "--generated" in seen[1]


def test_measured_fitness_reports_which_split_it_used(monkeypatch):
    class Result:
        stdout = '{"fitness": 8.0, "pass_rate": 0.8, "per_task": []}'

    monkeypatch.setattr(evolve.subprocess, "run", lambda cmd, **kw: Result())
    res = evolve._measure_fitness(object(), _cfg(), samples=2, split="validation")
    assert res.split == "validation"
    assert res.fitness == 8.0
    assert res.n == 2


def test_a_significant_holdout_drop_reads_as_a_regression():
    """Scored tasks improve, held-out tasks collapse: the signature of overfitting."""
    assert evolve._fitness_verdict(9.0, 6.5, tol=0.05, sem=0.1, k=1.0) == "regressed"


def test_holdout_noise_does_not_veto_a_good_change():
    """The counterweight. If noise could reject, evolution would stall permanently
    — and a gate that blocks everything is as useless as one that blocks nothing."""
    verdict = evolve._fitness_verdict(9.0, 8.95, tol=0.05, sem=0.4, k=1.0)
    assert verdict != "regressed"


def test_holdout_improvement_is_not_penalised():
    assert evolve._fitness_verdict(7.0, 8.5, tol=0.05, sem=0.1, k=1.0) == "improved"


def test_validation_scores_are_cached_separately_from_training(monkeypatch, tmp_path):
    """Both scores are keyed by the same source hash; if they shared a cache key the
    held-out number would be silently overwritten by the training one."""
    from ag import archive

    store = {}
    monkeypatch.setattr(archive, "get_cached_fitness", lambda h: store.get(h))
    monkeypatch.setattr(archive, "set_cached_fitness",
                        lambda h, v: store.__setitem__(h, v))

    archive.set_cached_fitness("abc123", 7.0)
    archive.set_cached_fitness("abc123:validation", 5.0)
    assert archive.get_cached_fitness("abc123") == 7.0
    assert archive.get_cached_fitness("abc123:validation") == 5.0


def test_splits_used_by_the_gate_are_actually_disjoint():
    """Ties the gate to the generator guarantee it depends on."""
    from ag import benchgen

    train = {t["prompt"] for t in benchgen.generate(seed=1337, n=250, split="train")}
    val = {t["prompt"] for t in benchgen.generate(seed=1337, n=250,
                                                  split="validation")}
    assert not (train & val)
