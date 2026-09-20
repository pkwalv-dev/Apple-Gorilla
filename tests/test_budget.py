"""Run budgets: explicit ceilings with a declared outcome.

An agent loop without a budget stops when the model decides to stop — for an
unattended autonomous run (the pentest-with-sub-agents case) that is the one
guaranteed way to burn GPU-hours on a task that will never converge. These pin
the ceiling semantics: what charges, what trips, and that "unlimited" really is.
"""
from __future__ import annotations

import dataclasses

import pytest

from ag.budget import Budget, BudgetExceeded
from ag.config import Config


def test_all_zeros_mean_unlimited():
    b = Budget()
    for _ in range(1000):
        b.charge_model_call(100, 100)
        b.charge_tool_call("calc")
    assert b.model_calls == 1000 and b.tool_calls == 1000


def test_model_call_ceiling_trips_with_a_named_reason():
    b = Budget(max_model_calls=2)
    b.charge_model_call(10, 10)
    b.charge_model_call(10, 10)
    with pytest.raises(BudgetExceeded) as e:
        b.charge_model_call(10, 10)
    assert "model-call" in e.value.reason and "2" in e.value.reason


def test_token_ceiling_tracks_in_and_out():
    b = Budget(max_tokens=100)
    b.charge_model_call(40, 20)          # 60
    with pytest.raises(BudgetExceeded):
        b.charge_model_call(30, 20)      # 110 > 100


def test_tool_call_ceiling_names_the_tool():
    b = Budget(max_tool_calls=1)
    b.charge_tool_call("calc")
    with pytest.raises(BudgetExceeded) as e:
        b.charge_tool_call("python_exec")
    assert "python_exec" in e.value.reason


def test_wall_clock_ceiling():
    b = Budget(wall_s=0.01)
    import time
    time.sleep(0.02)
    with pytest.raises(BudgetExceeded) as e:
        b.check_wall()
    assert "wall-clock" in e.value.reason


def test_from_config_is_none_when_everything_unlimited():
    cfg = dataclasses.replace(Config(), budget_model_calls=0, budget_tool_calls=0,
                              budget_tokens=0, budget_wall_s=0.0)
    assert Budget.from_config(cfg) is None


def test_from_config_reads_the_fields():
    cfg = dataclasses.replace(Config(), budget_model_calls=5, budget_wall_s=60.0)
    b = Budget.from_config(cfg)
    assert b.max_model_calls == 5 and b.wall_s == 60.0


def test_summary_reports_spend():
    b = Budget()
    b.charge_model_call(100, 20)
    b.charge_tool_call("calc")
    s = b.summary()
    assert "1 model call" in s and "1 tool call" in s and "120 token" in s
