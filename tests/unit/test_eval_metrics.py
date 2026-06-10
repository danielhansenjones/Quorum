from __future__ import annotations

from quorum.eval.classification import axis_metrics, compare_to_baseline, refusal_metrics
from quorum.eval.cost_report import summarize_trace_rows

# ---- axis metrics ----


def test_axis_perfect() -> None:
    pred = [{"profitability"}, {"growth", "leverage"}]
    exp = [{"profitability"}, {"growth", "leverage"}]
    m = axis_metrics(pred, exp)
    assert m["exact_set_match"] == 1.0
    assert m["per_axis"]["profitability"]["f1"] == 1.0
    assert m["per_axis"]["growth"]["f1"] == 1.0


def test_axis_partial() -> None:
    # Predicted growth missing on case 2, and an extra leverage on case 1.
    pred = [{"profitability", "leverage"}, set()]
    exp = [{"profitability"}, {"growth"}]
    m = axis_metrics(pred, exp)
    assert m["exact_set_match"] == 0.0
    assert m["per_axis"]["leverage"]["fp"] == 1
    assert m["per_axis"]["growth"]["fn"] == 1
    assert m["per_axis"]["profitability"]["tp"] == 1


def test_axis_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        axis_metrics([{"growth"}], [])


# ---- refusal metrics ----


def test_refusal_metrics() -> None:
    pred = [True, True, False, False]
    exp = [True, False, False, True]
    m = refusal_metrics(pred, exp)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert m["accuracy"] == 0.5


# ---- regression baseline (10f) ----


def _result(macro_f1: float, refusal_precision: float) -> dict:
    return {
        "axis": {
            "macro_f1": macro_f1,
            "exact_set_match": 0.85,
            "per_axis": {
                "profitability": {"f1": 0.91},
                "growth": {"f1": 0.87},
                "leverage": {"f1": 0.89},
                "risk_factors": {"f1": 1.0},
            },
        },
        "refusal": {"precision": refusal_precision, "recall": 1.0, "accuracy": 1.0},
    }


_BASELINE = {
    "tolerance": 0.03,
    "metrics": {
        "axis.macro_f1": 0.918,
        "axis.per_axis.growth.f1": 0.87,
        "refusal.precision": 1.0,
    },
}


def test_baseline_no_regression_within_tolerance() -> None:
    gate = compare_to_baseline(_result(0.90, 1.0), _BASELINE)  # 0.918 -> 0.90 is 0.018 < tol
    assert gate["ok"] is True
    assert gate["regressions"] == []


def test_baseline_flags_drop_beyond_tolerance() -> None:
    gate = compare_to_baseline(_result(0.918, 0.80), _BASELINE)  # refusal 1.0 -> 0.80
    assert gate["ok"] is False
    assert [r["metric"] for r in gate["regressions"]] == ["refusal.precision"]


# ---- cost aggregation ----


def _row(rid: str, node: str, cost: float, t_in: int = 0, t_out: int = 0, cache: int = 0) -> dict:
    return {
        "request_id": rid,
        "node_name": node,
        "tokens_in": t_in,
        "tokens_out": t_out,
        "cache_read_tokens": cache,
        "cost_dollars_billed": cost,
    }


def test_summarize_empty() -> None:
    assert summarize_trace_rows([])["requests"] == 0


def test_summarize_two_requests() -> None:
    rows = [
        _row("r1", "llm:analyst", 0.02, t_in=2000, t_out=500),
        _row("r1", "llm:synthesizer", 0.01, t_in=1000, t_out=300, cache=200),
        _row("r2", "llm:analyst", 0.04, t_in=4000, t_out=800),
    ]
    s = summarize_trace_rows(rows)
    assert s["requests"] == 2
    assert abs(s["totals"]["cost"] - 0.07) < 1e-9
    assert abs(s["per_request"]["cost_mean"] - 0.035) < 1e-9
    # r1 cost 0.03, r2 cost 0.04
    assert abs(s["per_node"]["llm:analyst"]["cost_total"] - 0.06) < 1e-9
    assert s["per_node"]["llm:analyst"]["rows"] == 2
    assert s["totals"]["cache_read_tokens"] == 200
    assert abs(s["totals"]["cache_read_fraction"] - 200 / 7000) < 1e-9


def test_summarize_percentiles_single() -> None:
    s = summarize_trace_rows([_row("r1", "n", 0.5)])
    assert s["per_request"]["cost_p50"] == 0.5
    assert s["per_request"]["cost_p95"] == 0.5
