"""Generated benchmark tasks — an unbounded, non-memorizable fitness function.

`bench.py` is AG's utility function, and `evolve` can only climb what that function
measures. With 14 fixed hand-written tasks, the ceiling is "answers these 14 prompts",
and once they pass, evolution has nothing left to optimise. Worse, a fixed suite is
memorizable: a LoRA run over AG's own history can learn the answers without learning
the capability, and the gate would happily report improvement.

This module replaces the constant with a *process*. Each generator is a pure function
of a seed that produces a task plus its programmatically-checkable answer, so:

- **The suite is unbounded.** Ask for 40 tasks or 4000; they are all new.
- **It cannot be memorised.** Different seed, different numbers, same skill.
- **It is reproducible.** The same seed gives the identical suite, which is what
  makes an A/B fitness comparison between two candidate patches meaningful — both are
  scored on exactly the same questions.
- **It has a held-out split.** Evolution sees the `train` seeds; adoption is
  confirmed on `validation` seeds it never optimised against, which is the standard
  defence against overfitting the metric.
- **Difficulty is a dial.** Tiers 1-3 let the suite stay discriminative as AG
  improves: a suite everything passes measures nothing.

Every answer is computed in Python here, never by a model, which preserves the
invariant the whole design rests on: no model ever grades the thing being optimised.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import string
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

# A generator takes a seeded Random and a difficulty tier, and returns
# (prompt, check, expect, category).
Generated = Tuple[str, str, object, str]
Generator = Callable[[random.Random, int], Generated]

REGISTRY: Dict[str, Generator] = {}


def generator(name: str) -> Callable[[Generator], Generator]:
    def deco(fn: Generator) -> Generator:
        REGISTRY[name] = fn
        return fn
    return deco


ONLY_NUMBER = "Reply with only the number."
ONLY_ANSWER = "Reply with only the answer, no explanation."


# --------------------------------------------------------------------------- #
# Arithmetic and numeric reasoning
# --------------------------------------------------------------------------- #

@generator("arith")
def _gen_arith(rng: random.Random, tier: int) -> Generated:
    if tier <= 1:
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        op = rng.choice(["+", "-", "*"])
    elif tier == 2:
        a, b = rng.randint(100, 999), rng.randint(11, 99)
        op = rng.choice(["*", "+", "-"])
    else:
        a, b = rng.randint(1000, 9999), rng.randint(100, 999)
        op = rng.choice(["*", "+"])
    value = {"+": a + b, "-": a - b, "*": a * b}[op]
    return (f"What is {a} {op} {b}? {ONLY_NUMBER}", "numeric", value, "arithmetic")


@generator("percent")
def _gen_percent(rng: random.Random, tier: int) -> Generated:
    price = rng.choice([20, 40, 50, 80, 120, 250, 400]) * (1 if tier < 3 else 3)
    pct = rng.choice([10, 20, 25, 50] if tier <= 1 else [5, 15, 25, 35, 40, 60])
    final = price * (100 - pct) / 100
    expect = int(final) if float(final).is_integer() else round(final, 2)
    return (f"An item costs ${price} and is marked {pct}% off. What is the final "
            f"price in dollars? {ONLY_NUMBER}", "numeric", expect, "arithmetic")


@generator("sequence")
def _gen_sequence(rng: random.Random, tier: int) -> Generated:
    kind = rng.choice(["arith", "geom"] if tier <= 1 else
                      ["arith", "geom", "square", "fib"])
    if kind == "arith":
        start, step = rng.randint(2, 20), rng.randint(2, 12)
        seq = [start + i * step for i in range(5)]
    elif kind == "geom":
        start, ratio = rng.randint(1, 5), rng.choice([2, 3])
        seq = [start * ratio ** i for i in range(5)]
    elif kind == "square":
        off = rng.randint(0, 3)
        seq = [(i + 1 + off) ** 2 for i in range(5)]
    else:
        a, b = rng.randint(1, 4), rng.randint(1, 5)
        seq = [a, b]
        for _ in range(4):
            seq.append(seq[-1] + seq[-2])
    shown, nxt = seq[:-1], seq[-1]
    return (f"What is the next number in the sequence "
            f"{', '.join(str(s) for s in shown)}, ...? {ONLY_NUMBER}",
            "numeric", nxt, "logic")


@generator("base_convert")
def _gen_base(rng: random.Random, tier: int) -> Generated:
    n = rng.randint(8, 63) if tier <= 1 else rng.randint(64, 4095)
    target = rng.choice(["binary", "hexadecimal"] if tier <= 1
                        else ["binary", "hexadecimal", "octal"])
    value = {"binary": bin(n)[2:], "hexadecimal": hex(n)[2:],
             "octal": oct(n)[2:]}[target]
    return (f"Convert the decimal number {n} to {target}. Reply with only the "
            f"digits, no prefix such as 0x or 0b.", "equals", value, "arithmetic")


@generator("unit_convert")
def _gen_units(rng: random.Random, tier: int) -> Generated:
    table = [("kilometres", "metres", 1000), ("hours", "minutes", 60),
             ("minutes", "seconds", 60), ("kilograms", "grams", 1000),
             ("litres", "millilitres", 1000), ("gigabytes", "megabytes", 1024),
             ("days", "hours", 24), ("weeks", "days", 7)]
    src, dst, factor = rng.choice(table)
    n = rng.randint(2, 12) if tier <= 1 else rng.randint(13, 400)
    return (f"How many {dst} are in {n} {src}? {ONLY_NUMBER}",
            "numeric", n * factor, "arithmetic")


@generator("modular")
def _gen_modular(rng: random.Random, tier: int) -> Generated:
    a = rng.randint(50, 500) if tier <= 1 else rng.randint(1000, 99999)
    m = rng.choice([3, 5, 7, 9, 11, 13])
    return (f"What is {a} mod {m}? {ONLY_NUMBER}", "numeric", a % m, "arithmetic")


@generator("gcd")
def _gen_gcd(rng: random.Random, tier: int) -> Generated:
    import math
    g = rng.choice([2, 3, 4, 6, 7, 12]) if tier <= 1 else rng.choice([8, 9, 14, 21])
    a, b = g * rng.randint(2, 20), g * rng.randint(2, 20)
    return (f"What is the greatest common divisor of {a} and {b}? {ONLY_NUMBER}",
            "numeric", math.gcd(a, b), "arithmetic")


# --------------------------------------------------------------------------- #
# String and text manipulation — instruction-following under an exact format
# --------------------------------------------------------------------------- #

_WORDS = ("orbit river candle puzzle meadow anchor lantern harbor velvet cactus "
          "marble thunder ribbon quartz willow saffron ember glacier nomad pebble "
          "cobalt fennel juniper walnut zephyr").split()


@generator("reverse")
def _gen_reverse(rng: random.Random, tier: int) -> Generated:
    word = rng.choice(_WORDS) if tier <= 1 else "".join(rng.sample(_WORDS, 2))
    return (f"Reverse the string {word!r}. Reply with only the reversed string.",
            "equals", word[::-1], "text")


@generator("count_char")
def _gen_count_char(rng: random.Random, tier: int) -> Generated:
    n = 2 if tier <= 1 else 4
    phrase = " ".join(rng.sample(_WORDS, n))
    letters = [c for c in set(phrase) if c.isalpha() and phrase.count(c) >= 1]
    ch = rng.choice(sorted(letters))
    return (f"How many times does the letter {ch!r} appear in the text "
            f"{phrase!r}? {ONLY_NUMBER}", "numeric", phrase.count(ch), "text")


@generator("word_count")
def _gen_word_count(rng: random.Random, tier: int) -> Generated:
    n = rng.randint(4, 8) if tier <= 1 else rng.randint(9, 20)
    phrase = " ".join(rng.choices(_WORDS, k=n))
    return (f"How many words are in this text? {phrase!r} {ONLY_NUMBER}",
            "numeric", n, "text")


@generator("sort_words")
def _gen_sort_words(rng: random.Random, tier: int) -> Generated:
    n = 4 if tier <= 1 else 7
    words = rng.sample(_WORDS, n)
    return (f"Sort these words alphabetically and reply with them comma-separated "
            f"with no spaces: {','.join(words)}",
            "equals", ",".join(sorted(words)), "text")


@generator("acronym")
def _gen_acronym(rng: random.Random, tier: int) -> Generated:
    n = 3 if tier <= 1 else 5
    words = rng.sample(_WORDS, n)
    acro = "".join(w[0] for w in words).upper()
    return (f"Make an acronym from the first letter of each word, in order, in "
            f"upper case: {' '.join(words)}. {ONLY_ANSWER}",
            "equals", acro, "text")


@generator("caesar")
def _gen_caesar(rng: random.Random, tier: int) -> Generated:
    word = rng.choice(_WORDS)
    shift = rng.randint(1, 5) if tier <= 1 else rng.randint(6, 20)
    out = "".join(chr((ord(c) - 97 + shift) % 26 + 97) for c in word)
    return (f"Apply a Caesar cipher with a shift of {shift} to the lowercase word "
            f"{word!r} (wrapping z to a). Reply with only the resulting string.",
            "equals", out, "text")


# --------------------------------------------------------------------------- #
# Sorting, sets, and small algorithms
# --------------------------------------------------------------------------- #

@generator("sort_numbers")
def _gen_sort_numbers(rng: random.Random, tier: int) -> Generated:
    n = 4 if tier <= 1 else (6 if tier == 2 else 9)
    hi = 99 if tier <= 2 else 999
    nums = rng.sample(range(1, hi), n)
    order = rng.choice(["ascending", "descending"])
    out = sorted(nums, reverse=(order == "descending"))
    return (f"Sort these numbers in {order} order, comma-separated with no spaces: "
            f"{','.join(str(x) for x in nums)}",
            "equals", ",".join(str(x) for x in out), "logic")


@generator("set_ops")
def _gen_set_ops(rng: random.Random, tier: int) -> Generated:
    a = sorted(rng.sample(range(1, 30), 5 if tier <= 1 else 8))
    b = sorted(rng.sample(range(1, 30), 5 if tier <= 1 else 8))
    op = rng.choice(["in both", "in the first list but not the second"])
    if op == "in both":
        out = sorted(set(a) & set(b))
    else:
        out = sorted(set(a) - set(b))
    if not out:
        out = []
    expect = ",".join(str(x) for x in out) if out else "none"
    return (f"List A is {','.join(map(str, a))} and list B is "
            f"{','.join(map(str, b))}. Which numbers are {op}? Reply with the "
            f"numbers in ascending order, comma-separated with no spaces, or the "
            f"word 'none' if there are none.", "equals", expect, "logic")


@generator("min_max")
def _gen_min_max(rng: random.Random, tier: int) -> Generated:
    n = 5 if tier <= 1 else 12
    nums = rng.sample(range(-50, 500), n)
    which = rng.choice(["largest", "smallest"])
    val = max(nums) if which == "largest" else min(nums)
    return (f"What is the {which} number in this list: "
            f"{', '.join(map(str, nums))}? {ONLY_NUMBER}", "numeric", val, "logic")


@generator("sum_list")
def _gen_sum(rng: random.Random, tier: int) -> Generated:
    n = 4 if tier <= 1 else 8
    hi = 50 if tier <= 1 else 400
    nums = [rng.randint(1, hi) for _ in range(n)]
    return (f"What is the sum of these numbers: {', '.join(map(str, nums))}? "
            f"{ONLY_NUMBER}", "numeric", sum(nums), "arithmetic")


# --------------------------------------------------------------------------- #
# Dates — a classic small-model weakness, and objectively checkable
# --------------------------------------------------------------------------- #

@generator("date_add")
def _gen_date_add(rng: random.Random, tier: int) -> Generated:
    base = date(2020, 1, 1) + timedelta(days=rng.randint(0, 2500))
    delta = rng.randint(1, 30) if tier <= 1 else rng.randint(31, 400)
    out = base + timedelta(days=delta)
    return (f"What date is {delta} days after {base.isoformat()}? Reply with only "
            f"the date in YYYY-MM-DD format.", "equals", out.isoformat(), "dates")


@generator("weekday")
def _gen_weekday(rng: random.Random, tier: int) -> Generated:
    d = date(2020, 1, 1) + timedelta(days=rng.randint(0, 2500))
    names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
             "sunday"]
    return (f"What day of the week is {d.isoformat()}? Reply with only the day "
            f"name.", "equals", names[d.weekday()], "dates")


@generator("date_diff")
def _gen_date_diff(rng: random.Random, tier: int) -> Generated:
    a = date(2020, 1, 1) + timedelta(days=rng.randint(0, 1200))
    b = a + timedelta(days=rng.randint(1, 100 if tier <= 1 else 900))
    return (f"How many days are there from {a.isoformat()} to {b.isoformat()}? "
            f"{ONLY_NUMBER}", "numeric", (b - a).days, "dates")


# --------------------------------------------------------------------------- #
# Structured output — the capability agentic use depends on most
# --------------------------------------------------------------------------- #

@generator("json_build")
def _gen_json_build(rng: random.Random, tier: int) -> Generated:
    key = rng.choice(["status", "count", "name", "total", "id"])
    if key in ("count", "total", "id"):
        value: object = rng.randint(1, 500)
    elif key == "status":
        value = rng.choice(["ok", "error", "pending"])
    else:
        value = rng.choice(_WORDS)
    return (f"Output a JSON object with a single key {key!r} whose value is "
            f"{json.dumps(value)}. Output only the JSON, with no code fence and no "
            f"commentary.", "json_key", [key, value], "format")


@generator("json_extract")
def _gen_json_extract(rng: random.Random, tier: int) -> Generated:
    data = {w: rng.randint(1, 99) for w in rng.sample(_WORDS, 4 if tier <= 1 else 8)}
    if tier >= 2:
        data["nested"] = {"inner": rng.randint(100, 999)}
    target = rng.choice([k for k in data if k != "nested"])
    blob = json.dumps(data)
    return (f"Given this JSON: {blob}\nWhat is the value of the key {target!r}? "
            f"{ONLY_NUMBER}", "numeric", data[target], "format")


@generator("csv_field")
def _gen_csv_field(rng: random.Random, tier: int) -> Generated:
    cols = ["id", "name", "score", "city"]
    rows = []
    for i in range(3 if tier <= 1 else 6):
        rows.append([str(i + 1), rng.choice(_WORDS), str(rng.randint(10, 99)),
                     rng.choice(["oslo", "lima", "cairo", "perth"])])
    row_i = rng.randrange(len(rows))
    col = rng.choice(["name", "score", "city"])
    col_i = cols.index(col)
    table = ",".join(cols) + "\n" + "\n".join(",".join(r) for r in rows)
    return (f"Given this CSV:\n{table}\nWhat is the {col} in the row where id="
            f"{rows[row_i][0]}? {ONLY_ANSWER}",
            "equals", rows[row_i][col_i], "format")


@generator("exact_words")
def _gen_exact_words(rng: random.Random, tier: int) -> Generated:
    n = rng.randint(3, 5) if tier <= 1 else rng.randint(6, 9)
    topic = rng.choice(["the sea", "a mountain", "a city at night", "an old book",
                        "a thunderstorm", "a quiet forest"])
    pattern = r"^\s*[A-Za-z]+(?:[ ][A-Za-z]+){" + str(n - 1) + r"}\s*$"
    return (f"Describe {topic} in exactly {n} words. Use only letters and single "
            f"spaces — no punctuation, no quotes, and no extra commentary.",
            "regex", pattern, "instruction")


@generator("list_format")
def _gen_list_format(rng: random.Random, tier: int) -> Generated:
    n = rng.randint(3, 5)
    start = rng.randint(2, 20)
    step = rng.randint(2, 9)
    nums = [start + i * step for i in range(n)]
    return (f"List the first {n} numbers of the sequence starting at {start} and "
            f"increasing by {step} each time. Reply comma-separated with no spaces.",
            "equals", ",".join(map(str, nums)), "instruction")


# The next four measure STRICT compliance — not getting the right answer, but
# expressing it under a tight output contract. That distinction matters because it
# is exactly where abliterated / merged models drift: they know the answer but wrap
# it in prose, echo the prompt, or ignore a case/quoting constraint. These tasks are
# trivially winnable (the oracle scores 10/10 on them), so a low category score is
# unambiguous evidence of an instruction-following problem, not a knowledge gap.

@generator("echo_exactly")
def _gen_echo_exactly(rng: random.Random, tier: int) -> Generated:
    words = rng.sample(["orbit", "lantern", "copper", "willow", "harbor", "cinder",
                        "meadow", "quartz", "sable", "tundra"], rng.randint(2, 4))
    n = rng.randint(100, 999)
    text = " ".join(words) + f" {n}"
    return (f"Reply with exactly this text, character for character, and nothing "
            f"else — no quotes, no commentary:\n{text}", "equals", text,
            "instruction")


@generator("json_only")
def _gen_json_only(rng: random.Random, tier: int) -> Generated:
    name = rng.choice(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"])
    a, b = rng.randint(2, 30), rng.randint(2, 30)
    canonical = json.dumps({"name": name, "total": a + b},
                           separators=(", ", ": "))
    return (f"Output ONLY a JSON object on one line, no prose and no code fences, "
            f"with exactly these keys: \"name\": the string \"{name}\", and "
            f"\"total\": the number {a} + {b}.", "equals", canonical, "instruction")


@generator("all_caps")
def _gen_all_caps(rng: random.Random, tier: int) -> Generated:
    words = rng.sample(["steady", "signal", "north", "bridge", "silent", "copper",
                        "harbor", "window", "anchor", "spruce"], rng.randint(3, 5))
    text = " ".join(words)
    return (f"Rewrite the following in ALL CAPITAL LETTERS, and output nothing "
            f"else:\n{text}", "equals", text.upper(), "instruction")


@generator("single_word")
def _gen_single_word(rng: random.Random, tier: int) -> Generated:
    q, ans = rng.choice([
        ("What is the capital of France?", "Paris"),
        ("What color do you get by mixing blue and yellow paint?", "green"),
        ("How many legs does a spider have? Answer with the digit.", "8"),
        ("What is the opposite of the word 'hot'?", "cold"),
        ("Which planet is known as the red planet?", "Mars"),
        ("What is 15 minus 6? Answer with the digit.", "9"),
    ])
    return (f"{q} Reply with exactly ONE word and no punctuation.",
            "equals", ans, "instruction")


# --------------------------------------------------------------------------- #
# Logic
# --------------------------------------------------------------------------- #

@generator("syllogism")
def _gen_syllogism(rng: random.Random, tier: int) -> Generated:
    a, b, c = rng.sample(["bloops", "razzies", "lazzies", "florps", "grimps",
                          "wugs", "zibs"], 3)
    valid = rng.random() < 0.5
    if valid:
        q = (f"If all {a} are {b} and all {b} are {c}, are all {a} necessarily "
             f"{c}?")
        ans = "yes"
    else:
        q = (f"If all {a} are {b} and some {b} are {c}, are all {a} necessarily "
             f"{c}?")
        ans = "no"
    return (f"{q} Answer with only yes or no.", "equals", ans, "logic")


@generator("comparison")
def _gen_comparison(rng: random.Random, tier: int) -> Generated:
    names = rng.sample(["ana", "beto", "caro", "dani", "eli"], 3)
    vals = rng.sample(range(10, 99), 3)
    pairs = dict(zip(names, vals))
    facts = []
    ordered = sorted(pairs.items(), key=lambda kv: -kv[1])
    facts.append(f"{ordered[0][0]} is taller than {ordered[1][0]}")
    facts.append(f"{ordered[1][0]} is taller than {ordered[2][0]}")
    rng.shuffle(facts)
    which = rng.choice(["tallest", "shortest"])
    ans = ordered[0][0] if which == "tallest" else ordered[2][0]
    return (f"{facts[0].capitalize()}. {facts[1].capitalize()}. Who is the "
            f"{which}? Reply with only the name.", "equals", ans, "logic")


@generator("parity")
def _gen_parity(rng: random.Random, tier: int) -> Generated:
    nums = [rng.randint(1, 200) for _ in range(5 if tier <= 1 else 10)]
    want = rng.choice(["even", "odd"])
    count = sum(1 for n in nums if (n % 2 == 0) == (want == "even"))
    return (f"How many numbers in this list are {want}: "
            f"{', '.join(map(str, nums))}? {ONLY_NUMBER}", "numeric", count,
            "logic")


# --------------------------------------------------------------------------- #
# Suite assembly
# --------------------------------------------------------------------------- #

@dataclass
class SuiteSpec:
    """How to build a suite — the whole thing is reproducible from these four."""

    seed: int = 1337
    n: int = 40
    tier: int = 2
    split: str = "train"          # train | validation

    def key(self) -> str:
        return f"{self.split}-{self.seed}-{self.n}-t{self.tier}"


# The split is a seed-space partition, not a sample of one pool: `train` and
# `validation` derive different RNG streams from the same base seed, so a task can
# never leak from one into the other, and both are reproducible forever.
_SPLIT_SALT = {"train": 0, "validation": 982_451_653}

# Fraction of the TASK space (not the seed space) reserved for validation.
_VALIDATION_SHARE = 0.25


def split_of(prompt: str) -> str:
    """Which split a task belongs to, decided by the task itself.

    Salting the RNG seed separates the *seeds* the two splits draw from, but not
    the *tasks* they produce: a generator with a small parameter space (say, a
    percentage discount with a few dozen combinations) emits the same prompt from
    different seeds, so identical tasks appeared in both splits. That silently
    defeats the generalisation gate — the held-out set is only evidence if it is
    genuinely held out.

    Hashing the prompt makes membership a property of the task, so disjointness
    holds by construction no matter how narrow a generator's parameter space is.
    blake2b (not `hash()`) because the assignment must be stable across processes;
    Python's string hash is randomised per interpreter, which would reshuffle the
    splits on every run and make the validation score incomparable over time.
    """
    digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 10_000
    return "validation" if bucket < int(_VALIDATION_SHARE * 10_000) else "train"


def generate(spec: Optional[SuiteSpec] = None, **kw) -> List[dict]:
    """Build a reproducible list of task dicts ready for `bench.Task.from_dict`."""
    spec = spec or SuiteSpec(**kw)
    salt = _SPLIT_SALT.get(spec.split, 0)
    names = sorted(REGISTRY)
    out: List[dict] = []
    seen: set = set()
    i = 0
    # Round-robin across generators so every capability is represented even when n
    # is small — a suite that is 80% arithmetic measures arithmetic, not AG.
    while len(out) < spec.n and i < spec.n * 20:
        name = names[i % len(names)]
        rng = random.Random(spec.seed + salt + i * 7919)
        tier = spec.tier
        if tier <= 0:     # 0 = mixed difficulty, which keeps the suite discriminative
            tier = 1 + (i % 3)
        try:
            prompt, check, expect, category = REGISTRY[name](rng, tier)
        except Exception:
            i += 1
            continue
        if prompt in seen:
            i += 1
            continue
        # Enforce the task-space partition: a prompt belongs to exactly one split,
        # decided by its own hash, so train and validation are provably disjoint.
        if spec.split in _SPLIT_SALT and split_of(prompt) != spec.split:
            i += 1
            continue
        seen.add(prompt)
        out.append({
            "id": f"{spec.split[:3]}-{name}-{i}",
            "prompt": prompt, "check": check, "expect": expect,
            "category": category, "weight": 1.0,
            "generator": name, "tier": tier,
        })
        i += 1
    return out


def write_tasks(path, spec: Optional[SuiteSpec] = None, **kw) -> int:
    """Materialise a suite as the JSONL `bench.load_tasks` already reads."""
    from pathlib import Path
    tasks = generate(spec, **kw)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks),
                 encoding="utf-8")
    return len(tasks)


def self_check(n_per_generator: int = 12) -> List[str]:
    """Verify every generator produces a task its own checker accepts.

    A generator whose stated answer does not pass `bench.score_answer` would inject
    an impossible task and permanently depress fitness — the benchmark would punish
    AG for the benchmark's own bug. This runs in the test suite so that can never
    ship.
    """
    from . import bench
    problems: List[str] = []
    for name, fn in sorted(REGISTRY.items()):
        for i in range(n_per_generator):
            rng = random.Random(9001 + i * 131)
            for tier in (1, 2, 3):
                try:
                    prompt, check, expect, category = fn(rng, tier)
                except Exception as e:
                    problems.append(f"{name}: raised {e!r}")
                    continue
                if not prompt or not str(prompt).strip():
                    problems.append(f"{name}: empty prompt")
                if check not in bench._CHECKERS:
                    problems.append(f"{name}: unknown checker {check!r}")
                    continue
                task = bench.Task(id=name, prompt=prompt, check=check,
                                  expect=expect, category=category)
                # The ideal answer is the expectation itself, rendered the way a
                # compliant model would write it.
                ideal = _ideal_answer(check, expect)
                if bench.score_answer(task, ideal) < 1.0:
                    problems.append(
                        f"{name} (tier {tier}): own answer {ideal!r} fails its "
                        f"{check} checker for expect={expect!r}")
    return problems


def _ideal_answer(check: str, expect) -> str:
    if check == "json_key":
        return json.dumps({expect[0]: expect[1]})
    if check == "regex":
        return _satisfy_regex(str(expect))
    if check == "contains_all":
        items = expect if isinstance(expect, list) else [expect]
        return " ".join(str(i) for i in items)
    return str(expect)


def _satisfy_regex(pattern: str) -> str:
    """Produce a string matching the small set of patterns our generators emit."""
    m = re.match(r"^\^\\s\*\[A-Za-z\]\+\(\?:\[ \]\[A-Za-z\]\+\)\{(\d+)\}\\s\*\$$",
                 pattern)
    if m:
        return " ".join(["word"] * (int(m.group(1)) + 1))
    m = re.match(r"^\^\[A-Za-z\]\+ \[A-Za-z\]\+ \[A-Za-z\]\+\$$", pattern)
    if m:
        return "one two three"
    return "x"


def categories() -> List[str]:
    """Every category the generated suite can produce (for per-category reporting)."""
    seen = set()
    for name, fn in REGISTRY.items():
        try:
            _, _, _, cat = fn(random.Random(1), 1)
            seen.add(cat)
        except Exception:
            continue
    return sorted(seen)


def info() -> dict:
    return {"generators": sorted(REGISTRY), "n_generators": len(REGISTRY),
            "categories": categories(), "splits": sorted(_SPLIT_SALT)}
