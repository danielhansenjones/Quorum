from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quorum.eval.runner import (
    CaseResult,
    GoldCase,
    _aggregate_judging,
    load_gold,
    run_all,
    run_case,
)
from quorum.state.citation import QuantCitation
from quorum.state.critique import Critique, FlaggedClaim


class _StreamGraph:
    # Mirrors the langgraph contract run_case relies on: stream_mode as a list
    # yields (mode, chunk) tuples; "updates" carries {node: update}, "values"
    # carries the accumulated state and the last one is final.
    def __init__(self, *, trajectory: list[str], final: dict[str, Any]) -> None:
        self._trajectory = trajectory
        self._final = final

    def stream(self, state: Any, stream_mode: Any = None) -> Any:
        yield ("values", {})
        for node in self._trajectory:
            yield ("updates", {node: {}})
        yield ("values", self._final)


def test_load_gold_yaml() -> None:
    root = Path(__file__).resolve().parents[2]
    cases = load_gold(root / "eval" / "datasets" / "v1" / "gold.yaml")
    assert len(cases) >= 5
    ids = {c.id for c in cases}
    assert "happy_aapl_msft_profitability" in ids
    assert "refuse_off_topic" in ids


def test_gold_case_distribution_floors() -> None:
    # Phase 10a target distribution. The seed file covers each category with
    # at least one case; the gate is ">=1 per category" for the seed and
    # ">=40 cases total" once the user expands it.
    root = Path(__file__).resolve().parents[2]
    cases = load_gold(root / "eval" / "datasets" / "v1" / "gold.yaml")
    by_status: dict[str, int] = {}
    for c in cases:
        by_status[c.expected_status] = by_status.get(c.expected_status, 0) + 1
    assert by_status.get("ok", 0) >= 1
    assert by_status.get("refused", 0) >= 1
    assert by_status.get("partial", 0) >= 1


def _case(case_id: str = "sim") -> GoldCase:
    return GoldCase(
        id=case_id,
        question="Q",
        expected_status="ok",
        expected_axes=["profitability"],
        expected_tickers=["AAPL", "MSFT"],
    )


def _flagged() -> FlaggedClaim:
    return FlaggedClaim(
        source_axis="profitability",
        claim="revenue grew 99%",
        flag="unsupported",
        reason="no supporting citation",
    )


def test_run_case_swallows_graph_exception() -> None:
    class _Graph:
        def stream(self, state: Any, stream_mode: Any = None) -> Any:
            raise RuntimeError("simulated")

    r = run_case(_case(), compiled_graph=_Graph())
    assert r.final_status == "error"
    assert "RuntimeError" in (r.error or "")
    # request_id is minted before the graph runs, so it survives a crash.
    assert r.request_id


def test_run_case_captures_critique_flagged_claims_and_turns() -> None:
    crit = Critique(status="ok", per_axis={}, flagged_claims=[_flagged()], turns_used=3)
    g = _StreamGraph(
        trajectory=["entry", "classify", "assess", "critic", "synthesize"],
        final={"report": "body", "status": "ok", "report_citations": [], "critique": crit},
    )
    r = run_case(_case(), compiled_graph=g)
    assert r.critique is not None
    assert r.critique["turns_used"] == 3
    assert r.critique["flagged_claims"][0]["claim"] == "revenue grew 99%"
    assert r.request_id


def test_run_case_captures_ordered_trajectory() -> None:
    traj = [
        "entry",
        "classify",
        "resolve",
        "plan",
        "analyze_axis",
        "assess",
        "critic",
        "synthesize",
    ]
    g = _StreamGraph(trajectory=traj, final={"report": "r", "status": "ok", "report_citations": []})
    r = run_case(_case(), compiled_graph=g)
    assert r.trajectory == traj


def test_run_all_persists_trajectory_and_critique_to_json(tmp_path: Path) -> None:
    gold = tmp_path / "gold.yaml"
    gold.write_text(
        "cases:\n"
        "  - id: c1\n"
        "    question: Q\n"
        "    expected_status: ok\n"
        "    expected_axes: [profitability]\n"
        "    expected_tickers: [AAPL, MSFT]\n"
    )
    crit = Critique(status="ok", per_axis={}, flagged_claims=[_flagged()], turns_used=2)
    g = _StreamGraph(
        trajectory=["entry", "assess", "critic", "synthesize"],
        final={"report": "b", "status": "ok", "report_citations": [], "critique": crit},
    )
    out = tmp_path / "run"
    run_all(gold, compiled_graph=g, out_dir=out)
    data = json.loads((out / "c1.json").read_text())
    assert data["trajectory"] == ["entry", "assess", "critic", "synthesize"]
    assert data["critique"]["turns_used"] == 2
    assert data["critique"]["flagged_claims"][0]["flag"] == "unsupported"
    assert data["request_id"]
    summary = json.loads((out / "summary.json").read_text())
    prov = summary["provenance"]
    assert set(prov) == {"git_sha", "git_dirty", "started_at", "judge_model"}
    assert prov["started_at"]


def _scored_case(
    case_id: str, *, status: str, turns: int, tool_calls: list[dict[str, Any]]
) -> CaseResult:
    r = CaseResult(
        case_id=case_id,
        request_id=f"req-{case_id}",
        final_status="ok",
        report="x",
        citations=[],
        elapsed_s=0.1,
        critique={
            "status": status,
            "turns_used": turns,
            "tool_calls": tool_calls,
            "flagged_claims": [],
        },
    )
    r.extra["scores"] = {
        "faithfulness": {"n": 1, "mean_score": 4.5, "faithful_fraction": 1.0},
        "quality": {"clarity": 5, "comparison_quality": 4, "notes": "x"},
        "incorporation": {"n": 2, "incorporated": 1, "rate": 0.5},
    }
    return r


def test_aggregate_judging_critic_block_and_tool_use_validity() -> None:
    bad = {"tool": "search_filings", "args": {"query": ""}, "ok": True, "result_summary": "s"}
    good = {
        "tool": "get_financial_concept",
        "args": {"ticker": "AAPL", "key": "profitability.revenue"},
        "ok": True,
        "result_summary": "s",
    }
    r1 = _scored_case("c1", status="ok", turns=3, tool_calls=[bad, good])
    r2 = _scored_case("c2", status="timeout", turns=5, tool_calls=[])
    critic = _aggregate_judging([r1, r2])["critic"]
    assert critic["cases_with_critique"] == 2
    assert critic["flagged_claims_total"] == 4
    assert critic["incorporation_rate"] == 0.5
    assert critic["turns_used_counts"] == {"3": 1, "5": 1}
    assert critic["timeout_rate"] == 0.5
    # The empty-query search_filings call is flagged; the concept call is clean.
    assert critic["tool_use"] == {"n": 2, "valid": 1, "valid_fraction": 0.5}


def test_aggregate_judging_critic_rebuttal_subblock() -> None:
    r = _scored_case("c1", status="ok", turns=2, tool_calls=[])
    r.report = "debt fell sharply this year."
    r.rebuttals = [
        {
            "source_axis": "profitability",
            "claim": "AAPL revenue tripled",
            "disposition": "retracted",
            "reason": "r",
        },
        {
            "source_axis": "leverage",
            "claim": "debt fell",
            "disposition": "defended",
            "reason": "r",
        },
    ]
    rb = _aggregate_judging([r])["critic"]["rebuttal"]
    assert rb["n"] == 2
    assert rb["retracted"] == 1
    assert rb["defended"] == 1
    assert rb["reflected"] == 2  # retracted claim absent, defended claim present
    assert rb["cases"] == 1


def test_aggregate_judging_critic_block_empty_without_critique() -> None:
    r = CaseResult(
        case_id="c1",
        request_id="req",
        final_status="refused",
        report="",
        citations=[],
        elapsed_s=0.0,
    )
    assert _aggregate_judging([r])["critic"] == {"cases_with_critique": 0}


def test_verify_quant_citation_deterministic_path() -> None:
    # No pool needed: the deterministic verifier short-circuits on empty results.
    from quorum.eval.judges import verify_quant_citation

    citation = QuantCitation(
        claim="x",
        ticker="AAPL",
        accession="acc",
        concept="profitability.revenue",
        value="383285000000",
        period="FY2025",
        unit="USD",
    )

    class _Pool:
        # The verifier calls get_financial_concept(pool, ...). Patch via the
        # module-level function. For unit test, simulate "no matching fact".
        def connection(self) -> Any:
            raise RuntimeError("not used in this path")

    # Stub get_financial_concept by monkey-patching its module attr.
    import quorum.eval.judges as judges_mod

    real = judges_mod.get_financial_concept
    judges_mod.get_financial_concept = lambda pool, ticker, key, periods=None: []  # type: ignore[assignment]
    try:
        verdict = verify_quant_citation(_Pool(), citation)  # type: ignore[arg-type]
        assert verdict.faithful is False
        assert verdict.judge == "deterministic"
        assert verdict.score == 1
    finally:
        judges_mod.get_financial_concept = real  # type: ignore[assignment]
