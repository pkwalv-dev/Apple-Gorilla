"""AG's fitness function — an objective, programmatic benchmark.

This is the ingredient that turns AG's `evolve` loop from a *blind* tuner (adopt
anything that keeps the tests green) into genuine **directed empirical
self-improvement**: a candidate self-edit is adopted only if it *measurably scores
at least as well* as the incumbent on a held-out task suite. That "keep-if-better"
selection is the shared core of every state-of-the-art self-improving system —
STOP's utility function (arXiv:2310.02304), the Darwin Gödel Machine's benchmark
validation (arXiv:2505.22954), and AlphaEvolve's automated evaluators
(arXiv:2506.13131).

Design invariants that make the fitness signal *sound*:

  1. **Verification is 100% programmatic — never model-judged.** Every task carries
     a deterministic checker (exact match / numeric / substring / regex / JSON).
     A model never grades the thing being optimised, so the loop cannot reward-hack
     by learning to flatter its own critic. This is the difference between a fitness
     function and a vibe.
  2. **Held-out and objective.** Tasks have a single verifiable answer, so a higher
     score can only come from actually answering more of them correctly.
  3. **Cheap and offline-capable.** Stdlib only. Runs against whatever backend is
     configured; on the dry-run stub it is a fast, deterministic smoke test.

Tasks live as data (embedded defaults + an optional `state/bench/tasks.jsonl`
override), so the suite itself is extensible — AG can grow its own benchmark as it
gains capabilities, which is what keeps the ceiling open-ended.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
from hashlib import sha256
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import Config, ROOT, STATE_DIR, ensure_dirs

BENCH_DIR = STATE_DIR / "bench"
TASKS_FILE = BENCH_DIR / "tasks.jsonl"

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Task:
    """One benchmark item: a prompt with a programmatically checkable answer."""

    id: str
    prompt: str
    check: str                       # equals | numeric | contains_all | regex | json_key
    expect: object                   # target value(s); shape depends on `check`
    category: str = "general"
    weight: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=str(d["id"]), prompt=str(d["prompt"]), check=str(d["check"]),
            expect=d.get("expect"), category=str(d.get("category", "general")),
            weight=float(d.get("weight", 1.0)),
        )


# --- the seed benchmark ----------------------------------------------------
# Objective, stable, model-answerable. Spans arithmetic, logic, recall, format,
# and instruction-following so a self-edit that helps one axis can't quietly wreck
# another. Keep answers terse so scoring is unambiguous and eval is cheap.
DEFAULT_TASKS: List[Task] = [
    Task("arith-mul", "What is 347 * 29? Reply with only the number.",
         "numeric", 10063, "arithmetic"),
    Task("arith-pow", "Compute 2 to the power of 15. Reply with only the number.",
         "numeric", 32768, "arithmetic"),
    Task("arith-div", "What is 1440 divided by 16? Reply with only the number.",
         "numeric", 90, "arithmetic"),
    Task("arith-pct", "A shirt costs $40 and is marked 25% off. What is the final "
         "price in dollars? Reply with only the number.", "numeric", 30, "arithmetic"),
    Task("logic-syllogism", "If all bloops are razzies and all razzies are lazzies, "
         "are all bloops necessarily lazzies? Answer with only yes or no.",
         "equals", "yes", "logic"),
    Task("logic-sort", "Sort these numbers in ascending order, comma-separated with "
         "no spaces: 5,3,9,1", "equals", "1,3,5,9", "logic"),
    Task("logic-sequence", "What is the next number in the sequence 2, 4, 8, 16, ...? "
         "Reply with only the number.", "numeric", 32, "logic"),
    Task("recall-gold", "What is the chemical symbol for gold? Reply with only the "
         "symbol.", "equals", "au", "recall"),
    Task("recall-apollo", "In what year did the Apollo 11 moon landing occur? Reply "
         "with only the year.", "numeric", 1969, "recall"),
    Task("recall-capital", "What is the capital city of Japan? Reply with only the "
         "city name.", "equals", "tokyo", "recall"),
    Task("recall-continents", "How many continents are there on Earth? Reply with "
         "only the number.", "numeric", 7, "recall"),
    Task("format-json", 'Output a JSON object with a single key "ok" whose value is '
         "the boolean true. Output only the JSON.", "json_key", ["ok", True], "format"),
    Task("format-three-words", "Reply with exactly three words separated by single "
         "spaces (no punctuation) describing the sea.", "regex",
         r"^[A-Za-z]+ [A-Za-z]+ [A-Za-z]+$", "format"),
    Task("instr-primes", "List the first three prime numbers, comma-separated with "
         "no spaces.", "equals", "2,3,5", "instruction"),
]


# --- verifiers (pure, deterministic, no model) -----------------------------

def _norm(s: str) -> str:
    """Normalise a free-text answer for exact comparison."""
    s = (s or "").strip().lower()
    # Models often wrap a terse answer in quotes/backticks or trail a period.
    s = s.strip("`\"'.,! \n\t")
    return s


def _last_number(s: str) -> Optional[float]:
    """The last number in the text — models tend to restate before concluding."""
    matches = _NUM.findall(s or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def score_answer(task: Task, answer: str) -> float:
    """Score one answer against a task's checker. Returns 1.0 (pass) or 0.0 (fail).

    Never raises: a malformed answer or checker simply fails closed to 0.0, so the
    fitness signal is always well-defined.
    """
    try:
        return _CHECKERS.get(task.check, lambda t, a: 0.0)(task, answer)
    except Exception:
        return 0.0


def _check_equals(task: Task, answer: str) -> float:
    return 1.0 if _norm(answer) == _norm(str(task.expect)) else 0.0


def _check_numeric(task: Task, answer: str) -> float:
    got = _last_number(answer)
    if got is None:
        return 0.0
    want = float(task.expect)
    # Relative tolerance for large values, absolute floor for small ones.
    return 1.0 if abs(got - want) <= max(1e-6, abs(want) * 1e-6) else 0.0


def _check_contains_all(task: Task, answer: str) -> float:
    hay = (answer or "").lower()
    needles = task.expect if isinstance(task.expect, list) else [task.expect]
    return 1.0 if all(str(n).lower() in hay for n in needles) else 0.0


def _check_regex(task: Task, answer: str) -> float:
    return 1.0 if re.search(str(task.expect), (answer or "").strip()) else 0.0


def _check_json_key(task: Task, answer: str) -> float:
    """expect == [key, value]: the answer must be JSON with that key set to value."""
    key, value = task.expect
    from .model import extract_json
    obj = extract_json(answer or "")
    return 1.0 if isinstance(obj, dict) and obj.get(key) == value else 0.0


_CHECKERS: Dict[str, Callable[[Task, str], float]] = {
    "equals": _check_equals,
    "numeric": _check_numeric,
    "contains_all": _check_contains_all,
    "regex": _check_regex,
    "json_key": _check_json_key,
}


# --- task loading ----------------------------------------------------------

def load_generated(cfg: Optional[Config] = None, *, split: str = "train",
                   n: Optional[int] = None, seed: Optional[int] = None,
                   tier: Optional[int] = None) -> List[Task]:
    """Build a suite from `ag/benchgen.py` — unbounded and non-memorizable.

    The hand-written `DEFAULT_TASKS` are a fixed 14 items: once AG passes them the
    fitness signal is saturated and evolution has nothing to climb. Generated tasks
    keep the signal alive indefinitely, and the `validation` split gives adoption a
    check that evolution never optimised against.
    """
    from . import benchgen
    spec = benchgen.SuiteSpec(
        seed=int(seed if seed is not None
                 else getattr(cfg, "bench_seed", 1337) if cfg else 1337),
        n=int(n if n is not None
              else getattr(cfg, "bench_generated_n", 40) if cfg else 40),
        tier=int(tier if tier is not None
                 else getattr(cfg, "bench_tier", 2) if cfg else 2),
        split=split)
    return [Task.from_dict(d) for d in benchgen.generate(spec)]


def load_tasks(limit: Optional[int] = None,
               cfg: Optional[Config] = None) -> List[Task]:
    """Return the benchmark tasks.

    Precedence: an explicit `state/bench/tasks.jsonl` (curated by a human) wins over
    everything; otherwise `bench_generated` selects the unbounded generated suite;
    otherwise the seed set. The curated file staying on top is deliberate — a human
    who wrote a task list meant it.
    """
    tasks = DEFAULT_TASKS
    if cfg is not None and getattr(cfg, "bench_generated", False) \
            and not TASKS_FILE.exists():
        gen = load_generated(cfg)
        return gen[:limit] if limit else gen
    try:
        if TASKS_FILE.exists():
            loaded: List[Task] = []
            for line in TASKS_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded.append(Task.from_dict(json.loads(line)))
                except Exception:
                    continue  # skip a corrupt line rather than failing the run
            if loaded:
                tasks = loaded
    except OSError:
        pass
    return tasks[:limit] if limit else list(tasks)


def tasks_hash(tasks: List[Task]) -> str:
    """Stable id for a task set, so cached fitness is invalidated if tasks change."""
    blob = json.dumps([asdict(t) for t in tasks], sort_keys=True)
    return sha256(blob.encode("utf-8")).hexdigest()[:16]


# --- running the benchmark -------------------------------------------------

@dataclass
class BenchResult:
    fitness: float = 0.0             # 0..10 weighted score (higher is better)
    pass_rate: float = 0.0           # fraction of tasks passed (0..1)
    n: int = 0
    passed: int = 0
    mode: str = "optimize_execute"
    elapsed_s: float = 0.0
    tasks_hash: str = ""
    per_task: List[dict] = field(default_factory=list)
    # Per-category pass rates. An aggregate score hides *where* AG is weak, and
    # "where" is exactly what the evolve briefing needs to aim a patch at something
    # specific rather than at the average.
    by_category: Dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def failed_ids(self) -> List[str]:
        return [t["id"] for t in self.per_task if not t["passed"]]

    @property
    def weakest_category(self) -> Optional[str]:
        """The category with the lowest pass rate — the sharpest evolution target."""
        if not self.by_category:
            return None
        return min(self.by_category.items(),
                   key=lambda kv: (kv[1]["pass_rate"], -kv[1]["n"]))[0]


def _answer_for(client, cfg: Config, task: Task, mode: str) -> str:
    """Produce AG's answer for one task, exercising the surface being optimised.

    - "execute": a single executor call (fast; measures the base model + executor
      system prompt).
    - "optimize_execute" (default): run the real prompt-optimizer first, then
      execute. This puts the *evolvable* optimizer/prompt files on the critical path
      so a prompt self-edit actually shows up in the fitness score.
    """
    from . import prompts
    from .pipeline import optimize
    if mode == "optimize_execute":
        eng_sys, eng_user, _ = optimize(client, cfg, task.prompt)
        res = client.complete(system=eng_sys, user=eng_user, cfg=cfg,
                              max_tokens=cfg.max_output_tokens)
        return res.text
    res = client.complete(system=prompts.EXECUTOR_SYSTEM_DEFAULT, user=task.prompt,
                          cfg=cfg, max_tokens=cfg.max_output_tokens)
    return res.text


def run_benchmark(client, cfg: Config, *, tasks: Optional[List[Task]] = None,
                  mode: Optional[str] = None, limit: Optional[int] = None,
                  emit=None, workers: Optional[int] = None) -> BenchResult:
    """Run the suite and return an aggregate fitness score (0..10).

    Fitness is the weighted mean of per-task pass scores, scaled to 0..10 to match
    AG's scorecard axes. Objective, repeatable, and cheap enough to gate every
    self-edit on.
    """
    from .pipeline import _emit
    ensure_dirs()
    mode = mode or getattr(cfg, "bench_mode", "optimize_execute")
    limit = limit if limit is not None else getattr(cfg, "bench_max_tasks", 0) or None
    tasks = tasks if tasks is not None else load_tasks(limit=limit, cfg=cfg)
    if workers is None:
        workers = int(getattr(cfg, "bench_workers", 1) or 1)
    t0 = time.time()
    per_task: List[dict] = []
    total_w = 0.0
    got_w = 0.0
    passed = 0

    def _run_one(idx_task):
        i, task = idx_task
        try:
            answer = _answer_for(client, cfg, task, mode)
        except Exception as e:  # a backend hiccup fails the task, never the run
            answer = f"(error: {e})"
        return i, task, answer

    # The benchmark is the dominant cost of an evolve cycle: bench_samples runs ×
    # tasks × (1-2 model calls each), all of them independent. Running them
    # concurrently is a pure wall-clock win — the tasks share no state, scoring is a
    # pure function, and the backend (Ollama or the Anthropic API) handles
    # concurrent requests. Kept at 1 by default because a single local GPU
    # serialises anyway and would only add queueing; raise it for an API backend or
    # a machine that can hold several requests in flight.
    results: List[tuple] = []
    if workers and workers > 1 and len(tasks) > 1:
        from concurrent.futures import ThreadPoolExecutor
        _emit(emit, "bench", f"running {len(tasks)} tasks on {workers} workers",
              level="tool")
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            for i, task, answer in pool.map(_run_one, enumerate(tasks)):
                results.append((i, task, answer))
    else:
        for i, task in enumerate(tasks):
            _emit(emit, "bench", f"task {i + 1}/{len(tasks)}: {task.id}", level="tool")
            results.append(_run_one((i, task)))

    # Score in task order regardless of completion order, so per_task is stable and
    # two runs of the same suite are directly comparable line by line.
    for i, task, answer in sorted(results, key=lambda r: r[0]):
        s = score_answer(task, answer)
        total_w += task.weight
        got_w += task.weight * s
        passed += 1 if s >= 1.0 else 0
        per_task.append({"id": task.id, "category": task.category,
                         "passed": s >= 1.0, "score": s,
                         "answer": (answer or "")[:200]})
    n = len(tasks)
    fitness = round(10.0 * (got_w / total_w), 3) if total_w else 0.0
    pass_rate = round(passed / n, 3) if n else 0.0
    cats: Dict[str, dict] = {}
    for t in per_task:
        c = cats.setdefault(t["category"], {"n": 0, "passed": 0, "pass_rate": 0.0})
        c["n"] += 1
        c["passed"] += 1 if t["passed"] else 0
    for c in cats.values():
        c["pass_rate"] = round(c["passed"] / c["n"], 3) if c["n"] else 0.0
    result = BenchResult(
        fitness=fitness, pass_rate=pass_rate, n=n, passed=passed, mode=mode,
        elapsed_s=round(time.time() - t0, 3), tasks_hash=tasks_hash(tasks),
        per_task=per_task, by_category=cats,
    )
    _emit(emit, "bench",
          f"fitness={fitness}/10  ({passed}/{n} passed) in {result.elapsed_s}s",
          level="result", scorecard={"fitness": fitness, "pass_rate": pass_rate})
    return result


# --- statistics of a nondeterministic fitness estimate ---------------------

@dataclass
class FitnessStat:
    """A fitness *estimate* with its uncertainty.

    Because the model is stochastic, one benchmark run is a single draw from a
    distribution. Repeating the run gives a sample mean whose uncertainty is the
    standard error (sem = stdev / sqrt(n)). The evolve gate compares means in units
    of sem, so it only reacts to differences that are unlikely to be noise.
    """

    mean: float = 0.0
    stdev: float = 0.0
    n: int = 0
    sem: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(samples: List[float]) -> FitnessStat:
    """Reduce repeated fitness measurements to (mean, stdev, n, standard error)."""
    vals = [float(s) for s in samples]
    n = len(vals)
    if n == 0:
        return FitnessStat()
    mean = sum(vals) / n
    stdev = statistics.stdev(vals) if n >= 2 else 0.0
    sem = stdev / (n ** 0.5) if n >= 2 else 0.0
    return FitnessStat(mean=round(mean, 3), stdev=round(stdev, 3), n=n,
                       sem=round(sem, 3))
