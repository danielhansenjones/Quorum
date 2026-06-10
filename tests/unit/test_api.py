from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from quorum.api.main import _summarize_node, create_app


def test_health_returns_ok() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_compare_requires_question() -> None:
    app = create_app(compiled_graph=object())  # any non-None to pass the guard
    client = TestClient(app)
    r = client.post("/compare", json={})
    assert r.status_code == 422


def test_compare_503_when_graph_missing() -> None:
    app = create_app(compiled_graph=None)
    client = TestClient(app)
    r = client.post("/compare", json={"question": "compare AAPL and MSFT"})
    assert r.status_code == 503


def test_ready_default_is_ok() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get("/ready")
    assert r.status_code == 200


def test_ready_503_when_check_fails() -> None:
    async def failing_check() -> dict[str, Any]:
        return {"ok": False, "checks": {"postgres": False}}

    app = create_app(ready_check=failing_check)
    client = TestClient(app)
    r = client.get("/ready")
    assert r.status_code == 503


def test_summarize_node_critic_flags_and_tools() -> None:
    from quorum.state.critique import Critique, FlaggedClaim, ToolCallRecord

    c = Critique(
        status="ok",
        per_axis={},
        flagged_claims=[
            FlaggedClaim(
                source_axis="growth",
                claim="revenue up sharply",
                flag="unsupported",
                reason="no year-over-year citation",
            )
        ],
        tool_calls=[
            ToolCallRecord(
                tool="search_filings",
                args={"query": "operating margin"},
                ok=True,
                result_summary="5 hits",
            )
        ],
        turns_used=3,
        duration_ms=1200,
    )
    out = _summarize_node("critic", {"critique": c})
    assert out["turns_used"] == 3
    assert out["flags"][0]["axis"] == "growth"
    assert out["tool_calls"][0]["tool"] == "search_filings"


def test_summarize_node_critic_none() -> None:
    assert _summarize_node("critic", {"critique": None}) == {"critique": None}


def test_summarize_node_analyze_axis_batched_fanout() -> None:
    from quorum.state.axis import AxisResult

    r = AxisResult(
        axis="profitability",
        mode="structured",
        per_company={},
        comparison="x",
        grounding="ok",
    )
    # The fan-out can batch parallel branch returns into a list under the node.
    out = _summarize_node("analyze_axis", [{"axis_results": [r]}])
    assert out["results"][0] == {
        "axis": "profitability",
        "grounding": "ok",
        "citations": 0,
        "error_kind": "none",
    }
