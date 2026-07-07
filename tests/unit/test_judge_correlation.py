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


def test_correlate_without_labels_has_no_ci() -> None:
    out = correlate([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0])
    assert out["n_clusters"] is None
    assert out["spearman_lower_95"] is None
    assert out["spearman_upper_95"] is None


def test_cluster_ci_counts_questions_not_rows() -> None:
    # Four claims from two questions: the CI resamples the two questions, so
    # n stays 4 but n_clusters reports the real unit of independence.
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 3.0, 4.0]
    labels = ["q1", "q1", "q2", "q2"]
    out = correlate(x, y, labels=labels)
    assert out["n"] == 4
    assert out["n_clusters"] == 2
    assert out["spearman_lower_95"] is not None
    assert out["spearman_lower_95"] <= out["spearman"] <= out["spearman_upper_95"]


def test_cluster_ci_is_deterministic() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [1.0, 2.0, 2.5, 4.0, 4.5, 6.0]
    labels = ["a", "a", "b", "b", "c", "c"]
    a = correlate(x, y, labels=labels, seed=0)
    b = correlate(x, y, labels=labels, seed=0)
    assert a["spearman_lower_95"] == b["spearman_lower_95"]


def test_decision_uses_local_when_both_lower_bounds_above_threshold() -> None:
    d = judge_decision(
        {"spearman": 0.9, "spearman_lower_95": 0.75, "n_clusters": 30},
        {"spearman": 0.8, "spearman_lower_95": 0.65, "n_clusters": 30},
    )
    assert d["use_local_for_iteration"] is True


def test_decision_falls_back_when_point_estimate_passes_but_ci_lower_fails() -> None:
    # Point estimate clears the bar; the wide lower bound does not. This is the
    # small-n case the gate exists to catch.
    d = judge_decision(
        {"spearman": 0.9, "spearman_lower_95": 0.8, "n_clusters": 30},
        {"spearman": 0.66, "spearman_lower_95": 0.1, "n_clusters": 7},
    )
    assert d["use_local_for_iteration"] is False


def test_decision_falls_back_when_ci_undefined() -> None:
    d = judge_decision(
        {"spearman": 0.9, "spearman_lower_95": None, "n_clusters": 2},
        {"spearman": 0.8, "spearman_lower_95": 0.7, "n_clusters": 30},
    )
    assert d["use_local_for_iteration"] is False
