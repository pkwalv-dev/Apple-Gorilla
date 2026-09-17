"""Tests for the directed-evolution scoring system (accuracy/quality/speed)."""
from ag import scoring
from ag.config import Config
from ag.model import make_client
from ag.pipeline import run as run_pipeline


def test_speed_score_full_marks_under_budget():
    assert scoring.speed_score(5.0, 200, budget_s=30.0) == 10.0
    # At the budget boundary -> still full marks.
    assert scoring.speed_score(30.0, 0, budget_s=30.0) == 10.0


def test_speed_score_decays_past_budget_never_negative():
    slow = scoring.speed_score(120.0, 0, budget_s=30.0)
    assert 0.0 < slow < 10.0
    # Absurdly slow trends toward 0 but never below.
    assert scoring.speed_score(10_000.0, 0, budget_s=1.0) >= 0.0


def test_speed_score_gives_large_outputs_more_time():
    # A big answer that takes longer isn't punished as hard as a tiny one would be.
    small = scoring.speed_score(60.0, 0, budget_s=30.0)
    big = scoring.speed_score(60.0, 6000, budget_s=30.0)
    assert big > small


def test_build_scorecard_weights_and_blend():
    card = scoring.build_scorecard(
        accuracy=10.0, quality=0.0, elapsed_s=1.0, output_tokens=100,
        budget_s=30.0, weights={"accuracy": 1.0, "quality": 0.0, "speed": 0.0},
    )
    # All weight on accuracy -> overall tracks accuracy.
    assert card.overall == 10.0
    assert card.speed == 10.0
    # answer_score excludes speed and here is pure accuracy.
    assert card.answer_score == 10.0


def test_build_scorecard_clamps_out_of_range():
    card = scoring.build_scorecard(
        accuracy=99.0, quality=-5.0, elapsed_s=1.0, output_tokens=10,
        budget_s=30.0,
    )
    assert card.accuracy == 10.0
    assert card.quality == 0.0


def test_weakest_axis_picks_lowest_average():
    cards = [
        {"accuracy": 9.0, "quality": 8.0, "speed": 3.0},
        {"accuracy": 8.0, "quality": 7.0, "speed": 4.0},
    ]
    assert scoring.weakest_axis(cards) == "speed"
    assert scoring.weakest_axis([]) == "accuracy"


def test_pipeline_scorecard_is_measured_speed_only():
    # The self-review loop is gone: a run reports MEASURED speed and leaves the
    # judged axes unscored (None) rather than fabricating them.
    cfg = Config()
    client = make_client(dry_run=True)
    rec = run_pipeline(client, cfg, "Explain entropy.")
    sc = rec.scorecard
    assert sc and "speed" in sc and sc["speed"] is not None
    assert sc.get("accuracy") is None and sc.get("quality") is None
    assert sc.get("overall") is None
    assert sc["elapsed_s"] >= 0.0
