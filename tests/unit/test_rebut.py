from __future__ import annotations

import json
from typing import Any

from quorum.graph.nodes.rebut import rebut
from quorum.state.axis import AxisResult
from quorum.state.citation import QuantCitation
from quorum.state.critique import FlaggedClaim


class _Cli:
    backend = "anthropic"
    model = "claude-sonnet-4-6"

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def chat(self, **kwargs: Any) -> Any:
        self.calls += 1

        class _Block:
            text = self._text

        class _Resp:
            content = [_Block()]

        return _Resp()


def _cite(concept: str) -> QuantCitation:
    return QuantCitation(
        claim="x",
        ticker="AAPL",
        accession="a",
        concept=concept,
        value="1",
        period="FY2024",
        unit="USD",
    )


def _axis_with_cites() -> AxisResult:
    return AxisResult(
        axis="profitability",
        mode="structured",
        per_company={},
        comparison="c",
        citations=[_cite("profitability.revenue"), _cite("profitability.gross_profit")],
        grounding="ok",
    )


def _fc(axis: str, claim: str) -> FlaggedClaim:
    return FlaggedClaim(source_axis=axis, claim=claim, flag="unsupported", reason="r")


def test_rebut_maps_dispositions_and_citations_and_decrements_budget() -> None:
    payload = json.dumps(
        {
            "rebuttals": [
                {"claim": "rev claim", "disposition": "defended", "reason": "ok", "cite_ref": "C1"},
                {
                    "claim": "margin claim",
                    "disposition": "retracted",
                    "reason": "no",
                    "cite_ref": None,
                },
            ]
        }
    )
    cli = _Cli(payload)
    out = rebut(
        flagged_claims=[_fc("profitability", "rev claim"), _fc("profitability", "margin claim")],
        axis_results=[_axis_with_cites()],
        remaining_steps=3,
        sonnet_client=cli,
    )
    assert out["remaining_steps"] == 2
    by_claim = {r.claim: r for r in out["rebuttals"]}
    assert by_claim["rev claim"].disposition == "defended"
    # "C1" resolves to the second (index 1) citation on that axis.
    assert by_claim["rev claim"].citation is not None
    assert by_claim["rev claim"].citation.concept == "profitability.gross_profit"
    assert by_claim["margin claim"].disposition == "retracted"
    assert by_claim["margin claim"].citation is None
    assert cli.calls == 1


def test_rebut_contains_analyst_error() -> None:
    class _Raise:
        backend = "anthropic"
        model = "m"

        def chat(self, **kwargs: Any) -> Any:
            raise RuntimeError("anthropic 503")

    out = rebut(
        flagged_claims=[_fc("profitability", "c")],
        axis_results=[_axis_with_cites()],
        remaining_steps=2,
        sonnet_client=_Raise(),
    )
    assert out["rebuttals"] == []
    assert out["remaining_steps"] == 1


def test_rebut_groups_by_axis_one_call_each() -> None:
    payload = json.dumps(
        {"rebuttals": [{"claim": "c", "disposition": "retracted", "reason": "r", "cite_ref": None}]}
    )
    cli = _Cli(payload)
    out = rebut(
        flagged_claims=[_fc("profitability", "c"), _fc("leverage", "c")],
        axis_results=[
            _axis_with_cites(),
            AxisResult(
                axis="leverage",
                mode="structured",
                per_company={},
                comparison="c",
                citations=[],
                grounding="ok",
            ),
        ],
        remaining_steps=4,
        sonnet_client=cli,
    )
    assert cli.calls == 2
    assert {r.source_axis for r in out["rebuttals"]} == {"profitability", "leverage"}


def test_rebut_skips_invalid_disposition() -> None:
    payload = json.dumps({"rebuttals": [{"claim": "c", "disposition": "maybe", "reason": "r"}]})
    out = rebut(
        flagged_claims=[_fc("profitability", "c")],
        axis_results=[_axis_with_cites()],
        remaining_steps=2,
        sonnet_client=_Cli(payload),
    )
    assert out["rebuttals"] == []
    assert out["remaining_steps"] == 1
