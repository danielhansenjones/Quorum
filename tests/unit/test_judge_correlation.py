from __future__ import annotations

from quorum.eval.judge_correlation import correlate, judge_decision


def test_correlate_monotonic_is_one() -> None:
    out = correlate([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert out["n"] == 4
    assert out["spearman"] == 1.0
    assert round(out["pearson"], 6) == 1.0


def test_correlate_constant_input_is_undefined() -> None:
    out = correlate([5.0, 5.0, 5.0, 5.0], [4.0, 3.0, 5.0, 2.0])
    assert out["pearson"] is None
    assert out["spearman"] is None
    assert out["note"] == "constant input"


def test_correlate_too_few_points() -> None:
    out = correlate([1.0, 2.0], [1.0, 2.0])
    assert out["note"] == "n<3"


def test_decision_uses_local_when_both_above_threshold() -> None:
    d = judge_decision({"spearman": 0.8}, {"spearman": 0.7})
    assert d["use_local_for_iteration"] is True


def test_decision_falls_back_when_quality_below_threshold() -> None:
    d = judge_decision({"spearman": 0.8}, {"spearman": 0.5})
    assert d["use_local_for_iteration"] is False


def test_decision_falls_back_when_correlation_undefined() -> None:
    d = judge_decision({"spearman": None}, {"spearman": 0.9})
    assert d["use_local_for_iteration"] is False
