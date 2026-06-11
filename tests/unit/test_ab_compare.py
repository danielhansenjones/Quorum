from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quorum.eval.ab_compare import compare_runs, load_run_dir, summarize_rebuttals
from quorum.eval.cost_report import per_request_cost

_QUALITY = {
    "clarity": 5,
    "comparison_quality": 4,
    "evidence_coverage": 4,
    "honesty": 5,
    "notes": "x",
}


def _write_case(
    d: Path,
    case_id: str,
    *,
    request_id: str,
    faith: float | None,
    incorporation: float | None = None,
    quality: dict[str, Any] | None = None,
    elapsed: float | None = None,
    report: str = "r",
    rebuttals: list[dict[str, Any]] | None = None,
) -> None:
    faithfulness = (
        {"n": 1, "mean_score": faith, "faithful_fraction": 1.0}
        if faith is not None
        else {"n": 0, "mean_score": None, "faithful_fraction": None}
    )
    scores: dict[str, Any] = {"faithfulness": faithfulness, "quality": quality}
    if incorporation is not None:
        scores["incorporation"] = {"n": 1, "incorporated": 1, "rate": incorporation}
    (d / f"{case_id}.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "request_id": request_id,
                "final_status": "ok",
                "elapsed_s": elapsed,
                "report": report,
                "citations": [],
                "rebuttals": rebuttals or [],
                "scores": scores,
            }
        )
    )


def _make_arm(root: Path, name: str, faiths: dict[str, float]) -> Path:
    d = root / name
    d.mkdir(parents=True)
    for cid, f in faiths.items():
        _write_case(
            d, cid, request_id=f"{name}-{cid}", faith=f, incorporation=0.5, quality=_QUALITY
        )
    (d / "summary.json").write_text(json.dumps({"n_cases": len(faiths)}))  # must be ignored
    return d


def test_compare_runs_computes_paired_deltas_and_ci(tmp_path: Path) -> None:
    a = _make_arm(tmp_path, "baseline", {"c1": 4.0, "c2": 4.0, "c3": 3.0, "c4": 5.0, "c5": 4.5})
    b = _make_arm(tmp_path, "critic", {"c1": 4.5, "c2": 4.2, "c3": 3.5, "c4": 5.0, "c5": 4.0})
    out = compare_runs(a, b)
    fb = out["metrics"]["faithfulness"]
    assert out["paired_cases"] == 5
    assert fb["n"] == 5
    assert abs(fb["mean_delta"] - 0.14) < 1e-9  # (0.5+0.2+0.5+0.0-0.5)/5
    lo, hi = fb["ci95"]
    assert lo <= fb["mean_delta"] <= hi
    # summary.json is not a case file.
    assert "summary" not in load_run_dir(a)


def test_compare_runs_is_deterministic(tmp_path: Path) -> None:
    a = _make_arm(tmp_path, "baseline", {"c1": 4.0, "c2": 3.0, "c3": 5.0, "c4": 4.2})
    b = _make_arm(tmp_path, "critic", {"c1": 4.5, "c2": 3.2, "c3": 4.0, "c4": 4.9})
    assert (
        compare_runs(a, b)["metrics"]["faithfulness"]["ci95"]
        == (compare_runs(a, b)["metrics"]["faithfulness"]["ci95"])
    )


def test_compare_runs_pairs_only_shared_cases(tmp_path: Path) -> None:
    a = _make_arm(tmp_path, "baseline", {"c1": 4.0, "c2": 4.0, "only_a": 3.0})
    b = _make_arm(tmp_path, "critic", {"c1": 4.5, "c2": 4.2, "only_b": 5.0})
    out = compare_runs(a, b)
    assert out["paired_cases"] == 2
    assert out["only_in_a"] == ["only_a"]
    assert out["only_in_b"] == ["only_b"]


def test_compare_runs_cost_delta_when_maps_supplied(tmp_path: Path) -> None:
    a = _make_arm(tmp_path, "baseline", {"c1": 4.0, "c2": 4.0})
    b = _make_arm(tmp_path, "critic", {"c1": 4.5, "c2": 4.2})
    # request_ids are "<arm>-<case>"; cost maps key on those.
    cost_a = {"baseline-c1": 0.02, "baseline-c2": 0.03}
    cost_b = {"critic-c1": 0.05, "critic-c2": 0.04}
    out = compare_runs(a, b, cost_a=cost_a, cost_b=cost_b)
    cost = out["metrics"]["cost"]
    assert cost["n"] == 2
    assert abs(cost["mean_delta"] - 0.02) < 1e-9  # ((0.05-0.02)+(0.04-0.03))/2


def test_summarize_rebuttals_counts_and_reflection() -> None:
    report = "MSFT margins rose strongly this year."  # defended present, retracted absent
    rebuttals = [
        {"disposition": "retracted", "claim": "AAPL revenue tripled"},
        {"disposition": "defended", "claim": "MSFT margins rose"},
    ]
    out = summarize_rebuttals(report, rebuttals)
    assert out["n"] == 2
    assert out["retracted"] == 1
    assert out["defended"] == 1
    assert out["reflected"] == 2
    assert out["reflected_rate"] == 1.0


def test_summarize_rebuttals_unreflected_when_retracted_claim_remains() -> None:
    out = summarize_rebuttals(
        "AAPL revenue tripled.", [{"disposition": "retracted", "claim": "AAPL revenue tripled"}]
    )
    assert out["reflected"] == 0
    assert out["reflected_rate"] == 0.0


def test_compare_runs_includes_rebuttal_block_and_latency(tmp_path: Path) -> None:
    a = tmp_path / "oneway"
    a.mkdir()
    b = tmp_path / "rebut"
    b.mkdir()
    _write_case(a, "c1", request_id="a1", faith=4.0, elapsed=1.0, report="X", rebuttals=[])
    _write_case(
        b,
        "c1",
        request_id="b1",
        faith=4.5,
        elapsed=1.5,
        report="clean report",
        rebuttals=[{"disposition": "retracted", "claim": "bad claim"}],
    )
    out = compare_runs(a, b)
    lat = out["metrics"]["latency"]
    assert lat["n"] == 1
    assert abs(lat["mean_delta"] - 0.5) < 1e-9
    rb_b = out["rebuttal"]["arm_b"]
    assert rb_b["n"] == 1
    assert rb_b["retracted"] == 1
    assert rb_b["reflected"] == 1  # "bad claim" absent from "clean report"
    assert out["rebuttal"]["arm_a"]["n"] == 0


def test_per_request_cost_sums_and_handles_null() -> None:
    rows = [
        {"request_id": "r1", "cost_dollars_billed": 0.01},
        {"request_id": "r1", "cost_dollars_billed": 0.02},
        {"request_id": "r2", "cost_dollars_billed": 0.05},
        {"request_id": "r3", "cost_dollars_billed": None},
    ]
    out = per_request_cost(rows)
    assert abs(out["r1"] - 0.03) < 1e-9
    assert out["r2"] == 0.05
    assert out["r3"] == 0.0
