"""Tests for AG's fitness function — the objective benchmark harness."""
import json

from ag import bench
from ag.bench import Task, score_answer, run_benchmark, load_tasks, tasks_hash
from ag.config import Config
from ag.model import ModelResult


# --- verifiers are pure and deterministic ----------------------------------

def test_equals_normalises_case_and_punctuation():
    t = Task("t", "p", "equals", "Tokyo")
    assert score_answer(t, "tokyo") == 1.0
    assert score_answer(t, "  TOKYO. ") == 1.0
    assert score_answer(t, "`Tokyo`") == 1.0
    assert score_answer(t, "Kyoto") == 0.0


def test_numeric_takes_last_number_with_tolerance():
    t = Task("t", "p", "numeric", 10063)
    assert score_answer(t, "The answer is 10063") == 1.0
    assert score_answer(t, "347*29 = 10063.0") == 1.0
    assert score_answer(t, "10062") == 0.0
    assert score_answer(t, "no number here") == 0.0


def test_contains_all_requires_every_needle():
    t = Task("t", "p", "contains_all", ["alpha", "beta"])
    assert score_answer(t, "Alpha and BETA present") == 1.0
    assert score_answer(t, "only alpha") == 0.0


def test_regex_checker():
    t = Task("t", "p", "regex", r"^[A-Za-z]+ [A-Za-z]+ [A-Za-z]+$")
    assert score_answer(t, "deep blue sea") == 1.0
    assert score_answer(t, "two words") == 0.0


def test_json_key_checker():
    t = Task("t", "p", "json_key", ["ok", True])
    assert score_answer(t, '{"ok": true}') == 1.0
    assert score_answer(t, 'here: {"ok": true} done') == 1.0
    assert score_answer(t, '{"ok": false}') == 0.0
    assert score_answer(t, "not json") == 0.0


def test_unknown_checker_fails_closed():
    t = Task("t", "p", "no_such_check", "x")
    assert score_answer(t, "anything") == 0.0


# --- the suite itself ------------------------------------------------------

def test_default_tasks_present_and_hash_stable():
    tasks = load_tasks()
    assert len(tasks) >= 10
    assert tasks_hash(tasks) == tasks_hash(load_tasks())  # deterministic


def test_load_tasks_uses_jsonl_override(monkeypatch, tmp_path):
    f = tmp_path / "tasks.jsonl"
    f.write_text(json.dumps({"id": "x", "prompt": "p", "check": "equals",
                             "expect": "y"}) + "\n")
    monkeypatch.setattr(bench, "TASKS_FILE", f)
    tasks = load_tasks()
    assert len(tasks) == 1 and tasks[0].id == "x"


class _OracleClient:
    """A perfect client: returns the exact expected answer for each task."""

    def __init__(self, answers):
        self._answers = answers

    def complete(self, *, system, user, cfg, max_tokens=None):
        for needle, ans in self._answers.items():
            if needle in user:
                return ModelResult(text=ans)
        return ModelResult(text="?")


def test_run_benchmark_perfect_client_scores_ten():
    # Two tiny tasks; an oracle answers both -> fitness 10, pass_rate 1.0.
    tasks = [Task("a", "TASK-A", "equals", "yes"),
             Task("b", "TASK-B", "numeric", 42)]
    client = _OracleClient({"TASK-A": "yes", "TASK-B": "42"})
    res = run_benchmark(client, Config(), tasks=tasks, mode="execute")
    assert res.fitness == 10.0
    assert res.pass_rate == 1.0
    assert res.passed == 2 and res.n == 2
    assert res.failed_ids == []


def test_run_benchmark_wrong_client_scores_zero():
    tasks = [Task("a", "TASK-A", "equals", "yes")]
    client = _OracleClient({"TASK-A": "no"})
    res = run_benchmark(client, Config(), tasks=tasks, mode="execute")
    assert res.fitness == 0.0
    assert res.failed_ids == ["a"]


def test_run_benchmark_partial_and_weighted():
    tasks = [Task("a", "TASK-A", "equals", "yes", weight=3.0),
             Task("b", "TASK-B", "equals", "yes", weight=1.0)]
    client = _OracleClient({"TASK-A": "yes", "TASK-B": "no"})
    res = run_benchmark(client, Config(), tasks=tasks, mode="execute")
    # 3 of 4 weight passed -> 7.5/10.
    assert res.fitness == 7.5
    assert res.passed == 1


def test_backend_error_fails_task_not_run():
    class Boom:
        def complete(self, *, system, user, cfg, max_tokens=None):
            raise RuntimeError("backend down")
    res = run_benchmark(Boom(), Config(),
                        tasks=[Task("a", "p", "equals", "x")], mode="execute")
    assert res.fitness == 0.0  # the run completes; the task just fails
    assert res.n == 1
