"""Self-improvement loop with a two-gate adoption test and instant rollback.

A self-edit is adopted only if it clears BOTH gates, and is rolled back in
milliseconds otherwise. AG never runs on unverified *or* regressive code:

  Gate 1 — SAFETY (does it still work?): the change must keep the test suite green.
  Gate 2 — FITNESS (is it actually better?): the change must score at least as well
           as the incumbent on AG's objective benchmark (ag/bench.py). This is the
           "keep-if-better" selection that STOP, the Darwin Gödel Machine, and
           AlphaEvolve all share — the difference between measured self-improvement
           and blind editing.

Flow:
  1. Measure the incumbent's fitness (cached by evolvable-files hash — free if
     unchanged since last time).
  2. Snapshot all evolvable files (fast, local).
  3. Ask the model for small patches, given recent telemetry AND which benchmark
     tasks currently fail (so evolution aims at a real gap).
  4. Apply patches to the working tree.
  5. Gate 1: run the test suite in a subprocess. Fail -> restore snapshot.
  6. Gate 2: re-measure fitness. A measured regression -> restore snapshot.
  7. Adopt (+ git-commit to AG's branch) and record the fitness delta to the
     evolution archive. Every kept change has a number attached, not a hope.

Autonomy levels (config.autonomy_level):
  - "never":   evolution disabled.
  - "manual":  propose + validate, but require --apply to keep (still auto-rolls
               back a failed/regressive candidate; a passing one is left for review).
  - "guarded": auto-adopt only if both gates pass; else auto-rollback (default).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import archive, backup, prompts
from .config import ROOT, Config
from .model import extract_json
from .pipeline import recent_runs


@dataclass
class EvolveResult:
    attempted: bool
    adopted: bool
    rolled_back: bool
    rationale: str = ""
    changed: List[str] = field(default_factory=list)
    test_output: str = ""
    snapshot_id: str = ""
    commit: Optional[str] = None
    reason: str = ""
    incumbent_fitness: Optional[float] = None
    candidate_fitness: Optional[float] = None
    fitness_delta: Optional[float] = None
    verdict: str = ""                 # improved | neutral | regressed | unknown
    candidate_stdev: Optional[float] = None   # spread across benchmark samples
    samples: int = 0                          # how many times fitness was measured
    margin: Optional[float] = None            # significance threshold actually used


def _fitness_verdict(incumbent: Optional[float], candidate: Optional[float], *,
                     tol: float, sem: float = 0.0, k: float = 1.0) -> str:
    """Classify a candidate's measured fitness against the incumbent's.

    Fitness is a *noisy* estimate (a stochastic model), so the decision margin is the
    LARGER of an absolute floor `tol` and `k` standard errors of the estimate — we
    only call a change real if it exceeds the noise. `sem` is the combined standard
    error of the incumbent/candidate means; with sem=0 this reduces exactly to the
    deterministic tolerance rule. Pure and side-effect free so it is unit-testable.

    A change within the margin is "neutral" (a safe lateral move — often a prompt
    clarification that helps real quality without shifting a small benchmark); a
    statistically-confident drop is "regressed" and must be rejected.
    """
    if incumbent is None or candidate is None:
        return "unknown"
    margin = max(float(tol), float(k) * float(sem))
    if candidate > incumbent + margin:
        return "improved"
    if candidate < incumbent - margin:
        return "regressed"
    return "neutral"


def _bench_once(client, cfg: Config) -> dict:
    """One benchmark run of the CURRENT on-disk source, in a SUBPROCESS.

    The subprocess is essential, not incidental: `evolve` patches files like
    `ag/prompts.py`, but this process already imported those modules, so an in-process
    benchmark would score the *old* prompts still held in memory. Shelling out to a
    fresh `python -m ag bench` guarantees the freshly-written candidate is measured —
    the same reason the test gate runs in a subprocess.
    """
    backend = {"ApiClient": "anthropic", "OllamaClient": "ollama",
               "DryRunClient": "dry"}.get(type(client).__name__)
    cmd = [sys.executable, "-m", "ag"]
    if backend:
        cmd += ["--backend", backend]
    cmd += ["bench", "--json", "--mode", getattr(cfg, "bench_mode", "optimize_execute")]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        return extract_json(r.stdout) or {}
    except (subprocess.TimeoutExpired, OSError):
        return {}


def _measure_fitness(client, cfg: Config, *, samples: Optional[int] = None, emit=None):
    """Estimate fitness by repeating the benchmark, since one run is a noisy draw.

    Runs the benchmark `samples` (default `cfg.bench_samples`) times and reduces the
    results to a mean and its standard error, so the evolve gate can decide on
    *statistical significance* rather than a single lucky/unlucky measurement. Each
    sample is an independent fresh process, so the model's nondeterminism is sampled,
    not frozen. Isolated behind one function so tests can stub it deterministically.

    Returns a lightweight object exposing `.fitness` (mean), `.stdev`, `.sem`, `.n`,
    `.samples`, and `.per_task` (from the last run, for the failing-task briefing).
    """
    from types import SimpleNamespace
    from . import bench
    from .pipeline import _emit
    n = samples if samples is not None else max(1, int(getattr(cfg, "bench_samples", 3)))
    fits: List[float] = []
    last: dict = {}
    for i in range(n):
        data = _bench_once(client, cfg)
        fits.append(float(data.get("fitness", 0.0) or 0.0))
        last = data or last
        if n > 1:
            _emit(emit, "bench", f"fitness sample {i + 1}/{n}: {fits[-1]}/10",
                  level="result")
    stat = bench.summarize(fits)
    return SimpleNamespace(
        fitness=stat.mean, stdev=stat.stdev, sem=stat.sem, n=stat.n, samples=fits,
        pass_rate=float(last.get("pass_rate", 0.0) or 0.0),
        per_task=list(last.get("per_task", []) or []),
    )


def _direction(cfg: Config, telemetry: list, bench_res=None) -> dict:
    """Summarise WHERE evolution should aim: the weakest quality axis across recent
    runs, the highest-friction wired tool, and — most concretely — which benchmark
    tasks currently FAIL. Failing tasks are the sharpest possible target: they point
    at a specific, verifiable capability gap. This is what makes the loop *directed*
    rather than a blind edit."""
    from . import inventory, scoring
    cards = [r.get("scorecard") for r in telemetry if r.get("scorecard")]
    inv = inventory.summary(cfg)
    axis = scoring.weakest_axis(cards) if cards else "accuracy"
    avg = {}
    if cards:
        for a in ("accuracy", "quality", "speed"):
            vals = [float(c.get(a, 0.0)) for c in cards]
            avg[a] = round(sum(vals) / len(vals), 2)
    briefing = {
        "weakest_score_axis": axis,
        "recent_axis_averages": avg,
        "highest_friction_wired_tool": inv["highest_friction_wired"],
        "avg_tool_friction": inv["avg_friction"],
        "hint": "Prefer prompt/principle/config edits that lift '" + axis
                + "'. Speed gains come from tighter prompts or fewer iterations; "
                "accuracy from sharper critic/executor guidance.",
    }
    if bench_res is not None:
        failing = [t for t in bench_res.per_task if not t["passed"]]
        briefing["benchmark_fitness"] = bench_res.fitness
        briefing["benchmark_pass_rate"] = bench_res.pass_rate
        briefing["failing_tasks"] = [
            {"id": t["id"], "category": t["category"],
             "your_answer": t.get("answer", "")[:120]} for t in failing[:6]
        ]
        briefing["fitness_hint"] = (
            "The strongest change makes a currently-FAILING benchmark task pass "
            "without breaking a passing one. Target the failing categories above via "
            "sharper executor/optimizer guidance in the evolvable prompt files."
        )
    return briefing


def _read_evolvable(cfg: Config) -> dict:
    files = {}
    for rel in cfg.evolvable_paths:
        p = ROOT / rel
        files[rel] = p.read_text() if p.exists() else ""
    return files


GATE_MISSING_MSG = (
    "pytest not installed — the self-improvement gate cannot verify changes. "
    "Install it: pip install pytest  (or pip install -r requirements.txt)"
)


def gate_available() -> bool:
    """True if the test gate can actually run.

    The suite relies on pytest fixtures (`tmp_path`, `monkeypatch`) and
    `pytest.raises`, so plain `unittest` cannot execute it — pytest is a hard
    requirement of the gate, not an optional accelerator. We detect it up front
    so the gate can fail CLOSED with an actionable message instead of silently
    rolling every candidate back on a cryptic "No module named pytest".
    """
    import importlib.util
    return importlib.util.find_spec("pytest") is not None


def run_tests() -> tuple[bool, str]:
    """Run the test suite in a subprocess. Passing tests gate adoption.

    Fails closed: if the gate can't run (pytest missing, no tests collected,
    timeout, crash), it reports failure so AG never adopts unverified code.
    """
    if not gate_available():
        return False, GATE_MISSING_MSG
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header"],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        # returncode 5 = "no tests collected"; treat as a failed gate, not a pass.
        if r.returncode == 5:
            return False, "no tests collected — refusing to adopt unverified change"
        return r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return False, "tests timed out"


def _validate_patch(rel: str, content: str) -> Optional[str]:
    """Cheap static validation before writing to disk. Returns error or None."""
    if rel.endswith(".json"):
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return f"invalid JSON: {e}"
    if rel.endswith(".py"):
        try:
            compile(content, rel, "exec")
        except SyntaxError as e:
            return f"syntax error: {e}"
    return None


def evolve(client, cfg: Config, *, apply: bool = False,
           dry_run: bool = False, emit=None, evolver_client=None) -> EvolveResult:
    """Attempt one gated self-improvement.

    `client` is the DEPLOY backend: it answers the benchmark, so fitness is always
    measured on the model you actually run. `evolver_client` (optional) is the model
    that PROPOSES the edit — pass a stronger one (e.g. Claude) to get smart mutations
    while the deploy model does the free, honest measuring. Defaults to `client`.
    """
    if cfg.autonomy_level == "never":
        return EvolveResult(False, False, False, reason="autonomy_level=never")

    # The proposer may differ from the deploy/measuring model; the benchmark never
    # runs on the proposer, so an adopted change is verified against `client`.
    proposer = evolver_client or client

    # No usable test gate means no safe way to verify a self-edit. Bail out BEFORE
    # spending a (possibly billed) model call, snapshotting, or touching the tree.
    if not gate_available():
        return EvolveResult(False, False, False,
                            reason="gate unavailable: " + GATE_MISSING_MSG)

    evolvable = _read_evolvable(cfg)
    parent_hash = archive.evolvable_hash(evolvable)
    telemetry = recent_runs(limit=5)

    # Fitness Gate, part 1 — measure the INCUMBENT before we touch anything, so the
    # candidate has a real bar to clear. Cached by the evolvable-files hash, so an
    # unchanged incumbent is scored at most once. Skipped in dry-run (the stub yields
    # an uninformative constant) and when the gate is disabled.
    do_fitness = bool(getattr(cfg, "fitness_gate", True)) and not dry_run
    incumbent = None
    incumbent_sem = None      # known only when we measure the incumbent fresh
    bench_res = None
    if do_fitness:
        incumbent = archive.get_cached_fitness(parent_hash)
        if incumbent is None:
            bench_res = _measure_fitness(client, cfg, emit=emit)
            incumbent = bench_res.fitness
            incumbent_sem = bench_res.sem
            archive.set_cached_fitness(parent_hash, incumbent)

    briefing = _direction(cfg, telemetry, bench_res=bench_res)

    user = (
        "# Directed-evolution briefing (aim your change here)\n"
        + json.dumps(briefing, indent=2)
        + "\n\n# Recent run telemetry (incl. accuracy/quality/speed scorecards)\n"
        + json.dumps(telemetry, indent=2)[:8000]
        + "\n\n# Current evolvable files\n"
        + json.dumps(evolvable, indent=2)[:12000]
        + "\n\nPropose the single small, safe change most likely to make a failing "
        "benchmark task pass (or lift the weakest axis) without breaking anything, "
        "per your rules."
    )
    res = proposer.complete(system=prompts.EVOLVER_SYSTEM, user=user, cfg=cfg,
                            max_tokens=cfg.meta_output_tokens)
    data = extract_json(res.text) or {}
    patches = data.get("patches", []) or []
    rationale = str(data.get("rationale", ""))

    if not patches:
        return EvolveResult(True, False, False, rationale=rationale,
                            reason="no patches proposed", incumbent_fitness=incumbent)

    # Keep only patches to declared evolvable paths, that pass static validation.
    valid = []
    for patch in patches:
        rel = patch.get("path", "")
        content = patch.get("new_content", "")
        if rel not in cfg.evolvable_paths:
            continue
        err = _validate_patch(rel, content)
        if err:
            return EvolveResult(True, False, False, rationale=rationale,
                                reason=f"rejected {rel}: {err}",
                                incumbent_fitness=incumbent)
        valid.append((rel, content))

    if not valid:
        return EvolveResult(True, False, False, rationale=rationale,
                            reason="no valid patches to evolvable paths",
                            incumbent_fitness=incumbent)

    # 1) Snapshot BEFORE touching anything, then bound how many we keep.
    snap = backup.snapshot(cfg.evolvable_paths, note=f"pre-evolve: {rationale[:80]}")
    backup.prune_snapshots(cfg.max_snapshots)

    # 2) Apply candidate patches.
    changed = []
    for rel, content in valid:
        (ROOT / rel).write_text(content)
        changed.append(rel)

    # 3) Gate 1 (SAFETY): the change must keep the test suite green.
    passed, output = run_tests()
    if not passed:
        backup.restore(snap)  # instant rollback
        archive.record(archive.Entry(
            ts=archive.now_ts(), parent_hash=parent_hash, candidate_hash="",
            incumbent_fitness=incumbent, candidate_fitness=None, delta=None,
            tests_passed=False, adopted=False, verdict="unknown", rationale=rationale,
            changed=changed, snapshot_id=snap.id))
        return EvolveResult(True, False, True, rationale=rationale, changed=changed,
                            test_output=output, snapshot_id=snap.id,
                            reason="tests failed -> rolled back",
                            incumbent_fitness=incumbent)

    # 4) Gate 2 (FITNESS): the change must not *significantly* regress the benchmark.
    # Fitness is a noisy estimate, so the decision uses the combined standard error of
    # the incumbent and candidate means — reacting only to differences beyond the
    # noise. combined_sem = sqrt(sem_inc^2 + sem_cand^2); if the incumbent came from
    # cache (no variance on hand), we assume its noise matches the candidate's.
    candidate = None
    verdict = "unknown"
    delta = None
    cand_stdev = None
    cand_n = 0
    margin = None
    candidate_hash = archive.evolvable_hash(_read_evolvable(cfg))
    if do_fitness:
        cand_res = _measure_fitness(client, cfg, emit=emit)
        candidate = cand_res.fitness
        cand_stdev = cand_res.stdev
        cand_n = cand_res.n
        archive.set_cached_fitness(candidate_hash, candidate)
        delta = round(candidate - incumbent, 3) if incumbent is not None else None
        inc_sem = incumbent_sem if incumbent_sem is not None else cand_res.sem
        combined_sem = math.hypot(inc_sem or 0.0, cand_res.sem or 0.0)
        k = float(getattr(cfg, "fitness_k", 1.0))
        margin = round(max(float(getattr(cfg, "fitness_tol", 0.05)), k * combined_sem), 3)
        verdict = _fitness_verdict(incumbent, candidate,
                                   tol=float(getattr(cfg, "fitness_tol", 0.05)),
                                   sem=combined_sem, k=k)
        if verdict == "regressed":
            backup.restore(snap)  # instant rollback — never adopt a real regression
            archive.record(archive.Entry(
                ts=archive.now_ts(), parent_hash=parent_hash,
                candidate_hash=candidate_hash, incumbent_fitness=incumbent,
                candidate_fitness=candidate, delta=delta, tests_passed=True,
                adopted=False, verdict=verdict, rationale=rationale, changed=changed,
                snapshot_id=snap.id, candidate_stdev=cand_stdev, samples=cand_n,
                margin=margin))
            return EvolveResult(
                True, False, True, rationale=rationale, changed=changed,
                test_output=output, snapshot_id=snap.id,
                reason=f"fitness regressed ({incumbent}->{candidate}, "
                       f"margin {margin}) -> rolled back",
                incumbent_fitness=incumbent, candidate_fitness=candidate,
                fitness_delta=delta, verdict=verdict, candidate_stdev=cand_stdev,
                samples=cand_n, margin=margin)

    # Passing candidate (both gates cleared).
    if cfg.autonomy_level == "manual" and not apply:
        # Leave the passing change in place for human review, but do not commit.
        return EvolveResult(
            True, False, False, rationale=rationale, changed=changed,
            test_output=output, snapshot_id=snap.id,
            reason="manual mode: passing candidate left for review "
                   "(use rollback to discard)",
            incumbent_fitness=incumbent, candidate_fitness=candidate,
            fitness_delta=delta, verdict=verdict, candidate_stdev=cand_stdev,
            samples=cand_n, margin=margin)

    # AG commits to its own branch, never main — advancing main stays a human action.
    commit = None if dry_run else backup.git_commit_evolve(
        f"AG self-improve (+{delta} fitness): {rationale[:48]}"
        if delta else f"AG self-improve: {rationale[:60]}",
        branch=cfg.evolve_branch,
    )
    archive.record(archive.Entry(
        ts=archive.now_ts(), parent_hash=parent_hash, candidate_hash=candidate_hash,
        incumbent_fitness=incumbent, candidate_fitness=candidate, delta=delta,
        tests_passed=True, adopted=True, verdict=verdict, rationale=rationale,
        changed=changed, snapshot_id=snap.id, candidate_stdev=cand_stdev,
        samples=cand_n, margin=margin))
    reason = "both gates passed -> adopted"
    if verdict == "improved":
        reason = f"fitness improved ({incumbent}->{candidate}, >{margin}) -> adopted"
    return EvolveResult(True, True, False, rationale=rationale, changed=changed,
                        test_output=output, snapshot_id=snap.id, commit=commit,
                        reason=reason, incumbent_fitness=incumbent,
                        candidate_fitness=candidate, fitness_delta=delta,
                        verdict=verdict, candidate_stdev=cand_stdev, samples=cand_n,
                        margin=margin)
