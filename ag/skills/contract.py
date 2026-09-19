"""The calling contract of a skill: which `args` keys its `run()` actually reads.

A skill is only usable if the model can guess its call correctly on the first try,
and the only thing the model has to go on is the example AG prints in the tool
description. So that example has to be derived from the code, not from what the
authoring model *said* about the code.

The failure this module exists to prevent: an authoring model asked for "the single
primary args key" answered `"celsius (float): the temperature in degrees Celsius"`.
That string was stored verbatim and rendered as the JSON key, so every live call
raised `KeyError: 'celsius'` — while the skill's own test, which calls
`run({'celsius': 0})` directly, passed. A skill can be born green and unusable,
and nothing downstream can tell.

`keys_read_by_run()` reads the keys out of the source with `ast`, which cannot be
talked out of the truth. `reconcile()` prefers those over the declaration and keeps
the declared one only when the code offers nothing to check it against.
"""
from __future__ import annotations

import ast
import re
from typing import List, Optional, Tuple

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# A declaration may be prose ("the temperature in Celsius"). These words are never a
# key, so a leading one means the declaration is a sentence, not an identifier.
_PROSE = frozenset({
    "a", "an", "the", "this", "that", "single", "primary", "main", "one", "first",
    "input", "value", "key", "arg", "args", "argument", "parameter", "param", "of",
    "for", "with", "and", "or", "to", "from", "it", "its", "run", "reads", "string",
})


def normalize_arg(declared: str) -> str:
    """Reduce a declared arg to a bare identifier, or "" if there isn't one.

    Handles the shapes authoring models actually emit: `"url"`, `"url (str)"`,
    `"url (str): the page to fetch"`, `'args["url"]'`, and plain prose (→ "").
    """
    text = (declared or "").strip()
    if not text:
        return ""
    # An explicit args["key"]/args['key'] reference names the key outright.
    m = re.search(r"""args\s*\[\s*['"]([^'"]+)['"]\s*\]""", text)
    if m:
        return m.group(1) if _IDENT.fullmatch(m.group(1)) else ""
    # Otherwise the key is the first identifier, before any type or description.
    head = re.split(r"[\s(:,\[\]{}=]", text.strip("\"'` "), maxsplit=1)[0]
    if _IDENT.fullmatch(head) and head.lower() not in _PROSE:
        return head
    return ""


def keys_read_by_run(code: str) -> List[str]:
    """The `args` keys `run()` reads, in source order, de-duplicated.

    Finds `args["k"]`, `args.get("k")`, and `args.pop("k")` against whatever the
    first parameter of `run` is named. Returns [] when the code reads no literal key
    (dynamic access, `**kwargs`-style use, or no `run` at all) — that is "unknown",
    not "none", and callers treat it that way.
    """
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "run"), None)
    if fn is None or not fn.args.args:
        return []
    bag = fn.args.args[0].arg          # conventionally "args", but read it
    out: List[str] = []

    def _add(k: object) -> None:
        if isinstance(k, str) and k and k not in out:
            out.append(k)

    for node in ast.walk(fn):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == bag and isinstance(node.slice, ast.Constant)):
            _add(node.slice.value)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in ("get", "pop")
              and isinstance(node.func.value, ast.Name)
              and node.func.value.id == bag and node.args
              and isinstance(node.args[0], ast.Constant)):
            _add(node.args[0].value)
    return out


def required_keys(code: str) -> List[str]:
    """The keys `run()` reads WITHOUT a default — the ones a caller must supply.

    `args["k"]` raises if absent; `args.get("k")` does not. Only the former is part
    of the promise the caller has to keep.
    """
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "run"), None)
    if fn is None or not fn.args.args:
        return []
    bag = fn.args.args[0].arg
    out: List[str] = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == bag and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and node.slice.value not in out):
            out.append(node.slice.value)
    return out


def reconcile(declared: str, code: str) -> Tuple[str, List[str], Optional[str]]:
    """Settle the contract: (primary key, all keys, error).

    The code is the authority. A declaration that names a key the code never reads is
    discarded, not honoured — publishing it would produce a call that cannot work.
    An error is returned only when no usable key can be established at all, which is
    a skill that should not be registered.
    """
    keys = keys_read_by_run(code)
    want = normalize_arg(declared)
    if keys:
        primary = want if want in keys else keys[0]
        # Put the primary first so the rendered example leads with it.
        ordered = [primary] + [k for k in keys if k != primary]
        return primary, ordered, None
    if want:
        # No literal key to check against (dynamic access): trust the declaration,
        # since the code demonstrably doesn't contradict it.
        return want, [want], None
    return "", [], ("skill declares no usable args key and run() reads none "
                    f"(declared {declared!r})")
