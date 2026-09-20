"""Generated benchmark tasks.

A self-improving system is only as good as its fitness function, so the
benchmark generator is load-bearing safety machinery. Two properties matter
above all: the expected answers must be COMPUTED (never model-authored), and
the validation split must be genuinely disjoint from the training split — an
overlap would silently turn the overfitting detector into a rubber stamp.
"""
from __future__ import annotations

from ag import benchgen


def test_self_check_finds_no_problems():
    """Every generator is executed and its own stated answer verified against its
    own checker. A generator whose expected answer fails its checker would make a
    task permanently unpassable and drag fitness down forever."""
    problems = benchgen.self_check()
    assert problems == [], f"generators disagree with their checkers: {problems}"


def test_generation_is_deterministic_for_a_seed():
    a = benchgen.generate(seed=99, n=25)
    b = benchgen.generate(seed=99, n=25)
    assert [t["prompt"] for t in a] == [t["prompt"] for t in b]


def test_different_seeds_give_different_tasks():
    a = {t["prompt"] for t in benchgen.generate(seed=1, n=25)}
    b = {t["prompt"] for t in benchgen.generate(seed=2, n=25)}
    assert a != b


def test_requested_count_is_honoured():
    for n in (1, 7, 40, 150):
        assert len(benchgen.generate(seed=5, n=n)) == n


def test_tasks_are_unique_within_a_suite():
    tasks = benchgen.generate(seed=7, n=200)
    prompts = [t["prompt"] for t in tasks]
    assert len(set(prompts)) == len(prompts), "duplicate tasks waste the budget"


def test_train_and_validation_splits_never_overlap():
    """The property that makes the generalisation gate meaningful."""
    train = {t["prompt"] for t in benchgen.generate(seed=1337, n=300, split="train")}
    val = {t["prompt"] for t in benchgen.generate(seed=1337, n=300, split="validation")}
    assert train and val
    assert not (train & val), "a shared task lets an overfit change pass Gate 3"


def test_every_task_is_well_formed():
    for t in benchgen.generate(seed=11, n=120, tier=0):
        assert t["prompt"].strip()
        assert t["check"] in ("equals", "numeric", "contains_all", "regex", "json_key")
        assert t["expect"] not in (None, "", [])
        assert t["category"]
        assert t["id"]


def test_task_ids_are_unique():
    ids = [t["id"] for t in benchgen.generate(seed=3, n=150)]
    assert len(set(ids)) == len(ids)


def test_categories_are_all_represented_in_a_large_suite():
    tasks = benchgen.generate(seed=21, n=120, tier=0)
    seen = {t["category"] for t in tasks}
    assert seen == set(benchgen.categories()), \
        "round-robin generation must cover every category"


def test_tiers_select_difficulty():
    for tier in (1, 2, 3):
        tasks = benchgen.generate(seed=4, n=20, tier=tier)
        assert tasks, f"tier {tier} produced nothing"


def test_expected_answers_are_computed_not_asserted():
    """Spot-check with an independent implementation. If the harness's own arithmetic
    were wrong, the whole fitness signal would be wrong in the same direction and no
    amount of evolution could reveal it."""
    checked = 0
    for t in benchgen.generate(seed=77, n=300, tier=0):
        pid = t["id"].split("-")[1] if "-" in t["id"] else ""
        if pid == "gcd":
            import math
            import re
            nums = [int(x) for x in re.findall(r"\d+", t["prompt"])]
            if len(nums) >= 2:
                assert int(t["expect"]) == math.gcd(nums[0], nums[1])
                checked += 1
        elif pid == "date_diff":
            import re
            from datetime import date
            ds = re.findall(r"(\d{4})-(\d{2})-(\d{2})", t["prompt"])
            if len(ds) == 2:
                d0 = date(*map(int, ds[0]))
                d1 = date(*map(int, ds[1]))
                assert int(t["expect"]) == abs((d1 - d0).days)
                checked += 1
    assert checked >= 2, "expected to independently verify at least a few tasks"


def test_write_tasks_roundtrips(tmp_path):
    import json

    out = tmp_path / "tasks.jsonl"
    n = benchgen.write_tasks(out, seed=5, n=20)
    assert n == 20
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 20
    assert all("prompt" in r and "expect" in r for r in rows)


def test_info_describes_the_generator_set():
    info = benchgen.info()
    assert info["n_generators"] > 20
    assert len(info["generators"]) == info["n_generators"]
    assert set(info["categories"]) == set(benchgen.categories())
    assert set(info["splits"]) == {"train", "validation"}


def test_generated_tasks_are_scorable_by_the_real_checkers():
    """End-to-end with bench.score_answer: a correct answer must score, and a wrong
    one must not. This is what closes the loop between generator and grader."""
    from ag import bench

    for spec in benchgen.generate(seed=31, n=60, tier=0):
        task = bench.Task.from_dict(spec)
        ideal = benchgen._ideal_answer(spec["check"], spec["expect"])
        assert bench.score_answer(task, ideal), \
            f"correct answer rejected for {spec['id']}: {spec['prompt'][:60]}"
        assert not bench.score_answer(task, "definitely not the answer 12345xyz"), \
            f"garbage accepted for {spec['id']}"


def test_split_membership_is_stable_across_processes():
    """Validation scores are compared over time, so a task must not drift between
    splits between runs. `hash()` would do exactly that (it is salted per process),
    which is why the assignment uses blake2b."""
    import subprocess
    import sys

    code = ("from ag import benchgen;"
            "print(benchgen.split_of('An item costs $120 and is marked 15% off.'))")
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert a.stdout.strip() == b.stdout.strip() != ""


def test_validation_share_is_a_meaningful_slice():
    """Too small a held-out set cannot detect a regression; too large starves
    the signal evolution optimises against."""
    # Measured on the UNFILTERED task space ("all" applies no split filter), which
    # is what the partition actually divides.
    prompts = [t["prompt"] for t in benchgen.generate(seed=8, n=400, tier=0, split="all")]
    share = sum(1 for p in prompts if benchgen.split_of(p) == "validation") / len(prompts)
    assert 0.1 < share < 0.45, f"held-out share {share:.2f} is not a useful slice"
