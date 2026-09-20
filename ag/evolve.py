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
from .pipeline import recent_runs, _emit


# --- verifier integrity -------------------------------------------------------
# The gates only protect AG while the thing they judge cannot edit the gates.
# config.json IS an evolvable file and cfg.evolvable_paths lives IN it, so without
# the checks below a candidate could (a) declare the gate files evolvable and edit
# them next cycle, (b) soften the safety knobs directly (`fitness_gate: false`,
# `fitness_k: 0`), or (c) change the measuring instrument the fitness subprocess
# reads from config.json (fewer bench tasks, a different model, an easier mode).
# These lists are literal code precisely so they sit OUTSIDE the reach of the
# system they constrain — configuration about safety is itself unsafe to evolve.

# Files that implement adoption, measurement, or authority. A candidate patch may
# never touch them, even if a (human- or candidate-written) evolvable_paths says so.
PROTECTED_FILES = frozenset({
    "ag/evolve.py", "ag/bench.py", "ag/benchgen.py", "ag/scoring.py",
    "ag/permissions.py", "ag/archive.py", "ag/backup.py", "conftest.py",
})

# config.json keys an evolve candidate may not change. Two families: the gate
# settings themselves, and every knob of the instrument the gate measures with —
# the bench subprocess reloads config.json, so a candidate that altered these
# would be re-grading itself on a different ruler.
SAFETY_CONFIG_KEYS = frozenset({
    # the gates
    "autonomy_level", "evolvable_paths", "fitness_gate", "bench_validate",
    "fitness_tol", "fitness_k", "bench_samples",
    # the measuring instrument
    "bench_mode", "bench_max_tasks", "bench_generated", "bench_generated_n",
    "bench_generated_tier", "bench_seed", "bench_workers",
    # which model answers / is measured
    "backend", "offline_backend", "model", "ollama_model", "ollama_host",
    "specialist_model", "ollama_options", "think", "effort",
    "max_output_tokens", "meta_output_tokens", "model_routing",
    # authority boundaries
    "allow_code_exec", "allow_external_tools", "allow_web", "allow_local_tools",
    "allow_acquire", "acquisition_autonomy", "skill_test_gate",
    "skill_full_suite_gate", "max_acquire_per_run", "github_allowlist",
})

REGRESSIONS_FILE = None  # resolved lazily from STATE_DIR (tests relocate state)


def _regressions_path():
    global REGRESSIONS_FILE
    if REGRESSIONS_FILE is None:
        from .config import STATE_DIR
        REGRESSIONS_FILE = STATE_DIR / "evolve" / "regressions.jsonl"
    return REGRESSIONS_FILE


def _gate_hashes() -> dict:
    """sha256 of every verifier file, for the pin-check across a candidate cycle."""
    import hashlib
    out = {}
    for rel in sorted(PROTECTED_FILES):
        p = ROOT / rel
        if p.exists():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _record_regression(gate: str, *, reason: str, changed=(), rationale: str = "",
                       incumbent=None, candidate=None) -> None:
    """Append one lived failure to the regression corpus.

    This corpus is how the loop LEARNS from its own rejections instead of
    re-proposing them: `_direction` feeds the most recent entries back into the
    proposer's briefing, so a gate failure becomes a lesson rather than a dead end.
    """
    try:
        path = _regressions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"when": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
               "gate": gate, "reason": reason[:400], "changed": list(changed)[:8],
               "rationale": rationale[:300],
               "incumbent_fitness": incumbent, "candidate_fitness": candidate}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass  # the corpus must never break the loop it observes


def regressions(limit: int = 20) -> List[dict]:
    """Most recent regression-corpus entries (newest last)."""
    try:
        path = _regressions_path()
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines[-limit:] if l.strip()]
    except Exception:
        return []


def _validate_config_patch(content: str) -> Optional[str]:
    """A candidate's config.json may not move the gates, the instrument, or the
    authority boundaries. Returns a rejection reason or None."""
    try:
        candidate = json.loads(content)
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}"
    try:
        incumbent = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    except Exception:
        incumbent = {}
    moved = sorted(k for k in SAFETY_CONFIG_KEYS
                   if k in candidate and candidate.get(k) != incumbent.get(k))
    if moved:
        return ("config change refused: " + ", ".join(moved) +
                " — these keys define the gate or the authority boundary and are "
                "human-set, not evolvable")
    return None


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


@dataclass
class ProposalPatch:
    """One candidate self-edit, surfaced for the human to accept or reject."""

    id: str
    path: str
    new_content: str
    valid: bool
    error: str = ""
    n_bytes: int = 0
    diff: str = ""

    def summary(self) -> dict:
        """The light view the GUI renders/selects on (omits full file content)."""
        return {"id": self.id, "path": self.path, "valid": self.valid,
                "error": self.error, "bytes": self.n_bytes, "diff": self.diff}


@dataclass
class ProposeResult:
    """Candidate self-edits from one propose() call — nothing has been applied."""

    attempted: bool
    reason: str
    rationale: str = ""
    parent_hash: str = ""
    directive: str = ""
    patches: List[ProposalPatch] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"attempted": self.attempted, "reason": self.reason,
                "rationale": self.rationale, "parent_hash": self.parent_hash,
                "directive": self.directive,
                "n_valid": sum(1 for p in self.patches if p.valid),
                "patches": [p.summary() for p in self.patches]}


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


def _validation_enabled(cfg: Config) -> bool:
    """Whether a held-out split exists to judge adoption on.

    Only meaningful with the generated suite: the 14 curated tasks are a fixed set
    with nothing to hold out. When enabled, `benchgen` partitions the *seed space*
    so validation tasks are provably disjoint from anything evolution optimises
    against.
    """
    return bool(getattr(cfg, "bench_validate", False)
                and getattr(cfg, "bench_generated", False))


def _bench_once(client, cfg: Config, *, split: str = "train") -> dict:
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
    if split == "validation":
        cmd += ["--generated", "--split", "validation"]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        return extract_json(r.stdout) or {}
    except (subprocess.TimeoutExpired, OSError):
        return {}


def _measure_fitness(client, cfg: Config, *, samples: Optional[int] = None, emit=None,
                     split: str = "train"):
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
        data = _bench_once(client, cfg, split=split)
        fits.append(float(data.get("fitness", 0.0) or 0.0))
        last = data or last
        if n > 1:
            _emit(emit, "bench", f"fitness sample {i + 1}/{n}: {fits[-1]}/10",
                  level="result")
    stat = bench.summarize(fits)
    return SimpleNamespace(
        fitness=stat.mean, stdev=stat.stdev, sem=stat.sem, n=stat.n, samples=fits,
        pass_rate=float(last.get("pass_rate", 0.0) or 0.0),
        per_task=list(last.get("per_task", []) or []), split=split,
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
            # Skip unscored axes (None, e.g. fast-mode runs) rather than treating
            # them as 0, which would misdirect evolution.
            vals = [float(c[a]) for c in cards if c.get(a) is not None]
            if vals:
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
    # The regression corpus is the loop's memory of what DIDN'T work — feeding it
    # back stops the proposer from re-attempting the same rejected edits cycle after
    # cycle. Failures become lessons, not dead ends.
    recent_failures = regressions(limit=4)
    if recent_failures:
        briefing["recent_rejected_proposals"] = [
            {"gate": r["gate"], "reason": r["reason"][:160]} for r in recent_failures
        ]
        briefing["regression_hint"] = (
            "These recent proposals were REJECTED by the gates. Do not re-propose "
            "them or close variants; aim somewhere the gates have not already vetoed."
        )
    return briefing


def _github_refs_block(cfg: Config, *, emit=None) -> str:
    """Fetch the configured vetted-GitHub reference files and format them for the
    proposer's briefing. Reference material only — evolve never executes it and still
    edits only evolvable_paths. Returns '' when disabled/empty/unreachable."""
    if not getattr(cfg, "evolve_use_github", False):
        return ""
    refs = getattr(cfg, "evolve_github_refs", None) or []
    if not refs:
        return ""
    from .permissions import PermissionBroker
    from .tools import github
    broker = PermissionBroker(allow_external_tools=True)
    broker.grant("github_fetch")
    allow = getattr(cfg, "github_allowlist", []) or []
    parts = []
    for r in refs[:5]:
        repo, path = str(r.get("repo", "")), str(r.get("path", ""))
        note = str(r.get("note", ""))
        if not (repo and path):
            continue
        text = github.fetch(repo, path, broker=broker, allowlist=allow,
                            ref=str(r.get("ref", "main")))
        if text.startswith("github_fetch error") or text.startswith("github_fetch refused"):
            continue
        _emit(emit, "evolve", f"reference: {repo}/{path}", level="tool")
        parts.append(f"## {repo}/{path}" + (f" — {note}" if note else "")
                     + f"\n```\n{text[:4000]}\n```")
    if not parts:
        return ""
    return ("\n\n# Vetted reference implementations (from your GitHub allowlist — adapt "
            "the ideas into the evolvable files; do NOT copy blindly, and only edit "
            "evolvable paths)\n" + "\n\n".join(parts))


def _read_evolvable(cfg: Config) -> dict:
    files = {}
    for rel in cfg.evolvable_paths:
        p = ROOT / rel
        files[rel] = p.read_text(encoding="utf-8") if p.exists() else ""
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
    if rel in PROTECTED_FILES:
        return ("rejected: this file implements the verification gates themselves "
                "and sits outside the evolvable surface")
    if rel == "config.json":
        err = _validate_config_patch(content)
        if err:
            return err
    elif rel.endswith(".json"):
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
           dry_run: bool = False, emit=None, evolver_client=None,
           directive: str = "") -> EvolveResult:
    """Attempt one gated self-improvement.

    `client` is the DEPLOY backend: it answers the benchmark, so fitness is always
    measured on the model you actually run. `evolver_client` (optional) is the model
    that PROPOSES the edit — pass a stronger one (e.g. Claude) to get smart mutations
    while the deploy model does the free, honest measuring. Defaults to `client`.
    `directive` (optional) is a free-text instruction from the user steering WHAT to
    improve this cycle; it aims the proposer but never bypasses the safety/fitness
    gates or the evolvable-path restriction.
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
    # Pin the verifier: capture the gate files' hashes now and re-verify after the
    # candidate is applied. A proposal cannot name these files (filtered above) and
    # cannot smuggle them in through config.json (checked in _validate_patch), so a
    # mismatch here means something outside the intended channel touched the gates.
    gate_hashes = _gate_hashes()

    # Fitness Gate, part 1 — measure the INCUMBENT before we touch anything, so the
    # candidate has a real bar to clear. Cached by the evolvable-files hash, so an
    # unchanged incumbent is scored at most once. Skipped in dry-run (the stub yields
    # an uninformative constant) and when the gate is disabled.
    do_fitness = bool(getattr(cfg, "fitness_gate", True)) and not dry_run
    incumbent = None
    incumbent_sem = None      # known only when we measure the incumbent fresh
    bench_res = None
    # Gate 3 uses a split the proposer never sees results from, so a change that
    # merely memorises the scored tasks cannot buy adoption with it.
    use_validation = do_fitness and _validation_enabled(cfg)
    incumbent_val = None
    incumbent_val_sem = None
    if do_fitness:
        incumbent = archive.get_cached_fitness(parent_hash)
        if incumbent is None:
            bench_res = _measure_fitness(client, cfg, emit=emit)
            incumbent = bench_res.fitness
            incumbent_sem = bench_res.sem
            archive.set_cached_fitness(parent_hash, incumbent)
    if use_validation:
        incumbent_val = archive.get_cached_fitness(parent_hash + ":validation")
        if incumbent_val is None:
            val_res = _measure_fitness(client, cfg, emit=emit, split="validation")
            incumbent_val = val_res.fitness
            incumbent_val_sem = val_res.sem
            archive.set_cached_fitness(parent_hash + ":validation", incumbent_val)
            _emit(emit, "bench", f"incumbent held-out fitness: {incumbent_val}/10",
                  level="result")

    briefing = _direction(cfg, telemetry, bench_res=bench_res)
    directive = (directive or "").strip()
    if directive:
        briefing["user_directive"] = directive
        _emit(emit, "evolve", f"user directive: {directive[:120]}", level="tool")

    directive_block = ""
    if directive:
        directive_block = (
            "\n\n# USER DIRECTIVE (highest priority — aim this cycle's change at "
            "satisfying this request, as long as it stays within your safety rules and "
            "the evolvable files; if it would require editing a non-evolvable file or "
            "weakening a gate, do the closest safe thing and say so in the rationale)\n"
            + directive)

    user = (
        "# Directed-evolution briefing (aim your change here)\n"
        + json.dumps(briefing, indent=2)
        + directive_block
        + _github_refs_block(cfg, emit=emit)
        + "\n\n# Recent run telemetry (incl. accuracy/quality/speed scorecards)\n"
        + json.dumps(telemetry, indent=2)[:8000]
        + "\n\n# Current evolvable files\n"
        + json.dumps(evolvable, indent=2)[:12000]
        + "\n\nPropose the single small, safe change most likely to "
        + ("satisfy the USER DIRECTIVE above (falling back to making a failing "
           "benchmark task pass or lifting the weakest axis) " if directive
           else "make a failing benchmark task pass (or lift the weakest axis) ")
        + "without breaking anything, per your rules."
    )
    # The proposer owes us JSON. A weak instruction-follower (the local abliterated
    # coder is a legitimate evolver backend) often wraps it in prose or drops the
    # contract entirely — so this call is verified-and-repaired, not assumed. If it
    # still can't comply, `data` is None and the cycle ends as "no patches", which
    # is an honest non-answer rather than a silently mis-parsed one.
    from .model import complete_json
    data, _raw, _attempts = complete_json(proposer, system=prompts.EVOLVER_SYSTEM,
                                          user=user, cfg=cfg, attempts=2,
                                          max_tokens=cfg.meta_output_tokens)
    data = data or {}
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
        (ROOT / rel).write_text(content, encoding="utf-8")
        changed.append(rel)

    # 2b) Verifier pin: the gate files must be byte-identical to the pre-candidate
    # capture, or the gates below would be vouching for a ruler that moved.
    if _gate_hashes() != gate_hashes:
        backup.restore(snap)
        moved = sorted(set(_gate_hashes()) ^ set(gate_hashes) |
                       {k for k in gate_hashes
                        if _gate_hashes().get(k) != gate_hashes[k]})
        _record_regression("integrity", reason=f"verifier files changed: {moved}",
                           changed=changed, rationale=rationale)
        return EvolveResult(True, False, True, rationale=rationale, changed=changed,
                            snapshot_id=snap.id,
                            reason="verifier files changed during the candidate "
                                   "cycle -> rolled back",
                            incumbent_fitness=incumbent)

    # 3) Gate 1 (SAFETY): the change must keep the test suite green.
    passed, output = run_tests()
    if not passed:
        backup.restore(snap)  # instant rollback
        _record_regression("tests",
                           reason="tests failed -> rolled back: " + output[-300:],
                           changed=changed, rationale=rationale,
                           incumbent=incumbent)
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
            _record_regression("fitness", changed=changed, rationale=rationale,
                               incumbent=incumbent, candidate=candidate,
                               reason=f"fitness regressed ({incumbent}->{candidate}, "
                                      f"margin {margin}) -> rolled back")
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

    # 5) Gate 3 (GENERALISATION): the candidate must also not regress on tasks that
    # were held out of the optimisation target. Gate 2 alone can be satisfied by a
    # change that fits the scored tasks; overfitting is the characteristic failure of
    # any keep-if-better loop, and the only honest detector is a split the loop does
    # not optimise against. Only a *significant* drop rejects, same statistics as
    # Gate 2 — noise must not veto a real improvement.
    val_fitness = None
    val_verdict = ""
    if use_validation:
        cand_val = _measure_fitness(client, cfg, emit=emit, split="validation")
        val_fitness = cand_val.fitness
        archive.set_cached_fitness(candidate_hash + ":validation", val_fitness)
        inc_val_sem = incumbent_val_sem if incumbent_val_sem is not None else cand_val.sem
        val_sem = math.hypot(inc_val_sem or 0.0, cand_val.sem or 0.0)
        val_verdict = _fitness_verdict(
            incumbent_val, val_fitness,
            tol=float(getattr(cfg, "fitness_tol", 0.05)),
            sem=val_sem, k=float(getattr(cfg, "fitness_k", 1.0)))
        _emit(emit, "bench",
              f"held-out fitness: {incumbent_val} -> {val_fitness} ({val_verdict})",
              level="result")
        if val_verdict == "regressed":
            backup.restore(snap)
            _record_regression("overfit", changed=changed, rationale=rationale,
                               incumbent=incumbent_val, candidate=val_fitness,
                               reason=f"held-out fitness regressed "
                                      f"({incumbent_val}->{val_fitness}) while the "
                                      f"scored split passed — overfitting")
            archive.record(archive.Entry(
                ts=archive.now_ts(), parent_hash=parent_hash,
                candidate_hash=candidate_hash, incumbent_fitness=incumbent,
                candidate_fitness=candidate, delta=delta, tests_passed=True,
                adopted=False, verdict="overfit", rationale=rationale,
                changed=changed, snapshot_id=snap.id, candidate_stdev=cand_stdev,
                samples=cand_n, margin=margin))
            return EvolveResult(
                True, False, True, rationale=rationale, changed=changed,
                test_output=output, snapshot_id=snap.id,
                reason=(f"held-out fitness regressed ({incumbent_val}->{val_fitness}) "
                        f"— change improved the scored tasks but not the held-out "
                        f"ones, so it was rolled back as overfitting"),
                incumbent_fitness=incumbent, candidate_fitness=candidate,
                fitness_delta=delta, verdict="overfit", candidate_stdev=cand_stdev,
                samples=cand_n, margin=margin)

    # Passing candidate (all gates cleared).
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


# --- Human-in-the-loop evolution -------------------------------------------
# The functions below replace AG's *automatic* adopt/reject decision with a
# curated one: propose() surfaces candidate edits (AG's own and any driven by the
# user's directive) without touching a file; the human picks which to keep; and
# apply_selected() applies exactly those, still enforcing the SAFETY test gate so a
# selection can never leave AG on code that fails its own suite.

def _diff_preview(rel: str, new_content: str, *, max_lines: int = 80) -> str:
    """A short unified diff of an evolvable file vs the proposed content, for review."""
    import difflib
    try:
        old = (ROOT / rel).read_text(encoding="utf-8")
    except OSError:
        old = ""
    lines = list(difflib.unified_diff(
        old.splitlines(), new_content.splitlines(),
        fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm=""))
    if not lines:
        return "(no textual change)"
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"… (+{len(lines) - max_lines} more diff lines)"]
    return "\n".join(lines)


def _evolver_prompt(cfg: Config, directive: str) -> str:
    """Build the proposer's user message (shared by propose() and evolve())."""
    telemetry = recent_runs(limit=5)
    evolvable = _read_evolvable(cfg)
    briefing = _direction(cfg, telemetry, bench_res=None)
    directive = (directive or "").strip()
    directive_block = ""
    if directive:
        briefing["user_directive"] = directive
        directive_block = (
            "\n\n# USER DIRECTIVE (highest priority — aim this cycle's change at "
            "satisfying this request, within your safety rules and the evolvable "
            "files; if it would require editing a non-evolvable file or weakening a "
            "gate, do the closest safe thing and say so in the rationale)\n" + directive)
    return (
        "# Directed-evolution briefing (aim your change here)\n"
        + json.dumps(briefing, indent=2)
        + directive_block
        + _github_refs_block(cfg)
        + "\n\n# Recent run telemetry (incl. accuracy/quality/speed scorecards)\n"
        + json.dumps(telemetry, indent=2)[:8000]
        + "\n\n# Current evolvable files\n"
        + json.dumps(evolvable, indent=2)[:12000]
        + "\n\nPropose small, safe change(s) most likely to "
        + ("satisfy the USER DIRECTIVE above (falling back to making a failing "
           "benchmark task pass or lifting the weakest axis) " if directive
           else "make a failing benchmark task pass (or lift the weakest axis) ")
        + "without breaking anything, per your rules. You may return more than one "
        "patch; each will be shown to a human who chooses which to keep.")


def propose(client, cfg: Config, *, evolver_client=None, directive: str = "",
            emit=None) -> ProposeResult:
    """Generate candidate self-edits and return them WITHOUT applying anything.

    This is the human-in-the-loop replacement for the automatic decision: no
    snapshot, no write, no adoption happens here. The caller shows the candidates
    and calls apply_selected() with the chosen subset.
    """
    if cfg.autonomy_level == "never":
        return ProposeResult(False, "autonomy_level=never")
    if not gate_available():
        return ProposeResult(False, "gate unavailable: " + GATE_MISSING_MSG)

    proposer = evolver_client or client
    parent_hash = archive.evolvable_hash(_read_evolvable(cfg))
    directive = (directive or "").strip()
    if directive:
        _emit(emit, "evolve", f"user directive: {directive[:120]}", level="tool")
    _emit(emit, "evolve", "asking the proposer for candidate change(s)…", level="tool")

    user = _evolver_prompt(cfg, directive)
    from .model import complete_json
    data, _raw, _attempts = complete_json(proposer, system=prompts.EVOLVER_SYSTEM,
                                          user=user, cfg=cfg, attempts=2,
                                          max_tokens=cfg.meta_output_tokens)
    data = data or {}
    raw_patches = data.get("patches", []) or []
    rationale = str(data.get("rationale", ""))

    patches: List[ProposalPatch] = []
    for i, patch in enumerate(raw_patches):
        rel = str((patch or {}).get("path", "")) or "(unspecified)"
        content = (patch or {}).get("new_content", "")
        content = content if isinstance(content, str) else str(content)
        if rel not in cfg.evolvable_paths:
            patches.append(ProposalPatch(
                id=f"p{i}", path=rel, new_content="", valid=False,
                error="not an evolvable file — AG may only edit its declared set"))
            _emit(emit, "evolve", f"proposed {rel} — rejected (not evolvable)",
                  level="error")
            continue
        err = _validate_patch(rel, content)
        patches.append(ProposalPatch(
            id=f"p{i}", path=rel, new_content=content, valid=(err is None),
            error=err or "", n_bytes=len(content.encode("utf-8")),
            diff=_diff_preview(rel, content)))
        _emit(emit, "evolve",
              f"proposed change to {rel}" + (f" — INVALID: {err}" if err else ""),
              level="result" if err is None else "error")

    n_valid = sum(1 for p in patches if p.valid)
    if not patches:
        reason = "no changes proposed"
    elif n_valid:
        reason = f"{n_valid} change(s) proposed — select which to apply"
    else:
        reason = "changes proposed but none are valid to apply"
    return ProposeResult(True, reason, rationale=rationale, parent_hash=parent_hash,
                         directive=directive, patches=patches)


def apply_selected(client, cfg: Config, selected, *, emit=None,
                   measure: bool = False, note: str = "") -> EvolveResult:
    """Apply a user-chosen subset of proposed patches, keeping the human's decision.

    `selected` is a list of {"path","new_content"} (as produced by propose()). The
    human already decided WHAT to keep by selecting, so there is no automatic
    fitness adopt/reject here. The SAFETY test gate is still enforced — a selection
    that breaks the suite is reverted, because that would break AG itself. Fitness is
    measured only when `measure=True`, and then purely as reported information.
    """
    if cfg.autonomy_level == "never":
        return EvolveResult(False, False, False, reason="autonomy_level=never")
    if not gate_available():
        return EvolveResult(False, False, False,
                            reason="gate unavailable: " + GATE_MISSING_MSG)

    # Re-validate defensively — never trust the selection to be well-formed or in-scope.
    valid = []
    for item in (selected or []):
        rel = str((item or {}).get("path", ""))
        content = (item or {}).get("new_content", "")
        if not isinstance(content, str) or rel not in cfg.evolvable_paths:
            continue
        if _validate_patch(rel, content):
            continue
        valid.append((rel, content))
    if not valid:
        return EvolveResult(True, False, False,
                            reason="no valid selected change(s) to apply")

    parent_hash = archive.evolvable_hash(_read_evolvable(cfg))
    rationale = (note or "").strip() or "user-selected change"

    incumbent = incumbent_sem = None
    do_fitness = bool(getattr(cfg, "fitness_gate", True)) and measure
    if do_fitness:
        incumbent = archive.get_cached_fitness(parent_hash)
        if incumbent is None:
            bres = _measure_fitness(client, cfg, emit=emit)
            incumbent = bres.fitness
            incumbent_sem = bres.sem
            archive.set_cached_fitness(parent_hash, incumbent)

    snap = backup.snapshot(cfg.evolvable_paths, note=f"pre-apply: {rationale[:80]}")
    backup.prune_snapshots(cfg.max_snapshots)
    changed = []
    for rel, content in valid:
        (ROOT / rel).write_text(content, encoding="utf-8")
        changed.append(rel)
    _emit(emit, "evolve",
          f"applied {len(changed)} selected change(s): " + ", ".join(changed),
          level="tool")

    # SAFETY gate stays hard: keeping code that fails the suite would break AG.
    passed, output = run_tests()
    if not passed:
        backup.restore(snap)
        archive.record(archive.Entry(
            ts=archive.now_ts(), parent_hash=parent_hash, candidate_hash="",
            incumbent_fitness=incumbent, candidate_fitness=None, delta=None,
            tests_passed=False, adopted=False, verdict="reverted",
            rationale=rationale, changed=changed, snapshot_id=snap.id))
        _emit(emit, "evolve", "safety tests FAILED — reverting selection", level="error")
        return EvolveResult(
            True, False, True, rationale=rationale, changed=changed,
            test_output=output, snapshot_id=snap.id,
            reason="reverted: the selected change(s) broke the safety test suite",
            incumbent_fitness=incumbent)

    # Fitness is INFORMATIONAL here — the human chose to keep, so we do not reject on it.
    candidate = delta = cand_stdev = None
    cand_n = 0
    candidate_hash = archive.evolvable_hash(_read_evolvable(cfg))
    if do_fitness:
        cres = _measure_fitness(client, cfg, emit=emit)
        candidate = cres.fitness
        cand_stdev = cres.stdev
        cand_n = cres.n
        archive.set_cached_fitness(candidate_hash, candidate)
        delta = round(candidate - incumbent, 3) if incumbent is not None else None

    commit = backup.git_commit_evolve(
        f"AG user-selected change: {rationale[:56]}", branch=cfg.evolve_branch)
    archive.record(archive.Entry(
        ts=archive.now_ts(), parent_hash=parent_hash, candidate_hash=candidate_hash,
        incumbent_fitness=incumbent, candidate_fitness=candidate, delta=delta,
        tests_passed=True, adopted=True, verdict="user-selected", rationale=rationale,
        changed=changed, snapshot_id=snap.id, candidate_stdev=cand_stdev,
        samples=cand_n))
    reason = "applied & kept — safety tests passed"
    if delta is not None:
        reason += f"; fitness {incumbent}->{candidate} (Δ{delta:+}, informational)"
    return EvolveResult(True, True, False, rationale=rationale, changed=changed,
                        test_output=output, snapshot_id=snap.id, commit=commit,
                        reason=reason, incumbent_fitness=incumbent,
                        candidate_fitness=candidate, fitness_delta=delta,
                        verdict="user-selected", candidate_stdev=cand_stdev,
                        samples=cand_n)
