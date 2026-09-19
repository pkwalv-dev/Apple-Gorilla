"""Local, offline tools — so AG can *compute and act*, not just summarize search.

These are the capabilities that make AG useful with the internet off: exact
arithmetic, reading and listing local files, running a short Python snippet, and
reading/writing AG's own memory. Each side-effecting tool is gated through the
PermissionBroker and off by default (see config.allow_local_tools); `calc` is pure
computation and needs no grant.

SECURITY: file access and code execution are powerful. They require an explicit
grant AND allow_external_tools, are bounded (output/time limits), and are never on
by default. `python_exec` runs real Python in a subprocess on this machine — only
enable it when you trust the prompt source.
"""
from __future__ import annotations

import ast
import operator
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..config import ROOT
from ..permissions import PermissionBroker

MAX_FILE_CHARS = 8000
MAX_EXEC_SECONDS = 10
MAX_OUTPUT_CHARS = 4000

# --- calc: safe arithmetic (no eval of arbitrary code) ---------------------

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calc(expr: str) -> str:
    """Evaluate a pure arithmetic expression exactly (+ - * / // % ** and parens).

    Fixes the small-model weakness at arithmetic. No permission needed — it cannot
    touch the machine.
    """
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
    except Exception as e:
        return f"calc error: {e}"
    return str(result)


# --- filesystem: gated read-only access ------------------------------------

def read_file(path: str, *, broker: PermissionBroker,
              max_chars: int = MAX_FILE_CHARS) -> str:
    broker.require("filesystem_read")
    p = Path(path).expanduser()
    try:
        data = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"read error: {e}"
    if len(data) > max_chars:
        return data[:max_chars] + f"\n…[truncated at {max_chars} chars]"
    return data


def list_dir(path: str = ".", *, broker: PermissionBroker) -> str:
    broker.require("filesystem_read")
    p = Path(path).expanduser()
    try:
        entries = sorted(p.iterdir())
    except OSError as e:
        return f"list error: {e}"
    lines = [(name.name + ("/" if name.is_dir() else "")) for name in entries[:200]]
    return "\n".join(lines) or "(empty)"


# --- code execution: gated subprocess --------------------------------------

def python_exec(code: str, *, broker: PermissionBroker,
                timeout: int = MAX_EXEC_SECONDS) -> str:
    """Run a short Python snippet in a subprocess; return its stdout/stderr.

    Gated by 'code_exec'. Bounded by a timeout and an output cap. Runs from the
    repo root with the same interpreter.
    """
    broker.require("code_exec")
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"exec error: timed out after {timeout}s"
    except Exception as e:  # pragma: no cover
        return f"exec error: {e}"
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    out = out.strip() or "(no output)"
    return out[:MAX_OUTPUT_CHARS]


# --- memory tools (thin wrappers so the reasoning loop can call them) -------

def memory_recall(query: str, *, k: int = 5) -> str:
    """Recall, annotated with how well-founded each item is. The reasoning loop is told
    which memories it may rely on and which are unconfirmed, rather than being handed a
    flat list it will read as fact."""
    from .. import memory
    mgr = memory.get_manager("root")
    hits = memory.recall(query, k=k)
    if not hits:
        return "(no relevant memories)"
    out = []
    for m in hits:
        b = mgr.belief(m)
        mark = "" if b >= mgr.trust_threshold else f" [unconfirmed, {b:.2f}]"
        if m.disputed:
            mark = f" [DISPUTED — a contradicting memory exists, {b:.2f}]"
        out.append(f"- {m.text}{mark}")
    return "\n".join(out)


def memory_remember(text: str, *, max_memories: int = 200) -> str:
    """Save a fact mid-reasoning. Stored as INFERENCE: this is the model asserting
    something, not the user stating it, and it must not be able to write itself a
    high-confidence belief. Corroboration from the user can raise it later."""
    from .. import memory
    mgr = memory.get_manager("root")
    m = memory.remember(text, max_memories=max_memories,
                        origin=memory.Origin.INFERENCE)
    if m is not None:
        return f"remembered (unconfirmed): {m.text}"
    # A refusal is worth saying out loud: the loop should learn that it cannot write
    # AG's self-description into memory, rather than silently retrying.
    return f"not stored: {mgr.last_refusal}" if mgr.last_refusal else "nothing to remember"
