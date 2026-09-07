"""Tests for the directed-evolution scoring system (accuracy/quality/speed)."""
from ag import scoring
from ag.config import Config
from ag.model import make_client
from ag.pipeline import run as run_pipeline, critique


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


def test_critique_falls_back_to_score_without_subscores():
    # The dry-run critic emits only `score`; accuracy/quality must fall back to it.
    cfg = Config()
    client = make_client(dry_run=True)
    crit = critique(client, cfg, "why is the sky blue?", "because rayleigh scattering")
    assert crit.accuracy == crit.score
    assert crit.quality == crit.score


def test_pipeline_populates_scorecard():
    cfg = Config()
    client = make_client(dry_run=True)
    # Full mode produces the judged+measured blend; fast mode (the default) leaves the
    # judged axes unscored and is covered in test_smoke.
    rec = run_pipeline(client, cfg, "Explain entropy.", fast=False)
    sc = rec.scorecard
    assert sc and set(("accuracy", "quality", "speed", "overall")).issubset(sc)
    assert 0.0 <= sc["overall"] <= 10.0
    assert sc["elapsed_s"] >= 0.0
