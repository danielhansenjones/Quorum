from __future__ import annotations

import json
import time
from typing import Any

from quorum.graph.nodes.critic import critic
from quorum.state.axis import AxisResult, CompanyAxisFinding


def _result(axis: str) -> AxisResult:
    return AxisResult(
        axis=axis,
        mode="structured",
        per_company={"AAPL": CompanyAxisFinding(ticker="AAPL")},
        comparison=f"comparison for {axis}",
        citations=[],
        grounding="ok",
        attempts=1,
    )


class _Block:
    def __init__(self, type: str, **kwargs: Any) -> None:
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, content: list[Any], stop_reason: str) -> None:
        self.content = content
        self.stop_reason = stop_reason


def test_critic_end_turn_with_valid_json() -> None:
    final = {
        "per_axis": {"profitability": {"groundedness": "ok", "notes": "looks fine"}},
        "cross_axis": [],
        "flagged_claims": [],
    }

    class Cli:
        backend = "anthropic"
        model = "claude-sonnet-4-6"

        def chat(self, **kwargs: Any) -> Any:
            return _Resp(
                content=[_Block("text", text=json.dumps(final))],
                stop_reason="end_turn",
            )

    c = critic(
        [_result("profitability")],
        sonnet_client=Cli(),
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        embed_query=lambda q: ([0.0], {}),
    )
    assert c is not None
    assert c.status == "ok"
    assert c.per_axis["profitability"].groundedness == "ok"
    assert c.turns_used == 1


def test_critic_exception_returns_none() -> None:
    class Cli:
        backend = "anthropic"
        model = "claude-sonnet-4-6"

        def chat(self, **kwargs: Any) -> Any:
            raise RuntimeError("anthropic 503")

    c = critic(
        [_result("profitability")],
        sonnet_client=Cli(),
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        embed_query=lambda q: ([0.0], {}),
    )
    # Containment property: any exception means critique = None for synthesize.
    assert c is None


def test_critic_caps_turns() -> None:
    # The critic keeps tool-calling indefinitely; the cap should force a fallback
    # Critique with empty flagged_claims after MAX_TURNS.
    class Cli:
        backend = "anthropic"
        model = "claude-sonnet-4-6"
        call_count = 0

        def chat(self, **kwargs: Any) -> Any:
            Cli.call_count += 1
            return _Resp(
                content=[
                    _Block(
                        "tool_use",
                        id=f"call-{Cli.call_count}",
                        name="search_filings",
                        input={"query": "x"},
                    )
                ],
                stop_reason="tool_use",
            )

    # Mock dispatch via embed_query path raising. We just need search_filings to
    # not blow up the loop; the dispatch's exception handling logs the tool_error.
    c = critic(
        [_result("profitability")],
        sonnet_client=Cli(),
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        embed_query=lambda q: ([0.0], {}),
        max_turns=3,
    )
    assert c is not None
    # We hit the cap without an end_turn JSON; status must reflect the fallback.
    assert c.turns_used == 3
    assert c.status in ("timeout", "failed", "partial")


def test_critic_wall_clock_timeout() -> None:
    class Cli:
        backend = "anthropic"
        model = "claude-sonnet-4-6"
        call_count = 0

        def chat(self, **kwargs: Any) -> Any:
            Cli.call_count += 1
            time.sleep(0.6)  # > wall_clock_s below per turn -> trigger timeout
            return _Resp(
                content=[
                    _Block(
                        "tool_use",
                        id=f"call-{Cli.call_count}",
                        name="search_filings",
                        input={"query": "x"},
                    )
                ],
                stop_reason="tool_use",
            )

    c = critic(
        [_result("profitability")],
        sonnet_client=Cli(),
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        embed_query=lambda q: ([0.0], {}),
        max_turns=5,
        wall_clock_s=0.5,
    )
    assert c is not None
    assert c.status in ("timeout", "failed", "partial")
