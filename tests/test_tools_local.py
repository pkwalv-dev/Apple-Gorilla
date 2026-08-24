"""Tests for the local, offline tool layer."""
import pytest

from ag.permissions import PermissionBroker, PermissionError_
from ag.tools import local


def _granted(*caps):
    b = PermissionBroker(allow_external_tools=True)
    for c in caps:
        b.grant(c)
    return b


def test_calc_exact_arithmetic():
    assert local.calc("(17*23)-4") == "387"
    assert local.calc("2**10") == "1024"
    assert local.calc("10/4") == "2.5"


def test_calc_rejects_code():
    # No names/calls/attributes — arbitrary code cannot run through calc.
    assert local.calc("__import__('os').system('x')").startswith("calc error")
    assert local.calc("open('x')").startswith("calc error")


def test_read_file_is_gated():
    b = PermissionBroker()  # no grant
    with pytest.raises(PermissionError_):
        local.read_file("whatever.txt", broker=b)


def test_read_file_and_list_dir_when_granted(tmp_path):
    b = _granted("filesystem_read")
    f = tmp_path / "note.txt"
    f.write_text("hello world")
    assert "hello world" in local.read_file(str(f), broker=b)
    listing = local.list_dir(str(tmp_path), broker=b)
    assert "note.txt" in listing


def test_read_file_truncates(tmp_path):
    b = _granted("filesystem_read")
    f = tmp_path / "big.txt"
    f.write_text("x" * 20000)
    out = local.read_file(str(f), broker=b, max_chars=100)
    assert "truncated" in out and len(out) < 200


def test_python_exec_is_gated():
    b = PermissionBroker()
    with pytest.raises(PermissionError_):
        local.python_exec("print(1)", broker=b)


def test_python_exec_runs_when_granted():
    b = _granted("code_exec")
    assert local.python_exec("print(sum(range(10)))", broker=b).strip() == "45"


def test_python_exec_times_out():
    b = _granted("code_exec")
    out = local.python_exec("while True: pass", broker=b, timeout=1)
    assert "timed out" in out
