from __future__ import annotations

import time
from typing import Any

import pytest

import quorum.graph.nodes.analyze_axis as az
from quorum.graph.nodes.analyze_axis import (
    _LegworkDispatch,
    _write_axis_result,
    analyze_axis_agentic,
)
from quorum.state.axis import AxisResult, AxisTask
from quorum.tools.concept_resolver import ResolvedFact
from quorum.tools.search import SearchHit


def _task(mode: str = "structured", tickers: tuple[str, ...] = ("AAPL", "MSFT")) -> AxisTask:
    return AxisTask(
        axis="profitability",
        mode=mode,  # type: ignore[arg-type]
        tickers=list(tickers),
        query_or_concept="profitability.revenue",
    )


class _FakeSonnet:
    backend = "anthropic"
    model = "claude-sonnet-4-6"

    def __init__(self, text: str) -> None:
        self._text = text

    def chat(self, **kwargs: Any) -> Any:
        class _Block:
            text = self._text

        class _Resp:
            content = [_Block()]

        return _Resp()


def _end_turn_client() -> Any:
    class _C:
        backend = "anthropic"
        model = "claude-haiku-4-5"

        def chat(self, **kwargs: Any) -> Any:
            class _Block:
                type = "text"
                text = "nothing to gather"

            class _Resp:
                content = [_Block()]
                stop_reason = "end_turn"

            return _Resp()

    return _C()


def _fact(concept: str, period: str) -> ResolvedFact:
    return ResolvedFact(
        value=1.0, unit="USD", period=period, accession="a", resolved_concept=concept
    )


def _hit(cid: str) -> SearchHit:
    return SearchHit(
        chunk_id=cid,
        score=1.0,
        payload={"chunk_id": cid, "ticker": "AAPL", "section": "item_1a", "text": "x"},
    )


def test_legwork_evidence_dedups_and_trims_quant() -> None:
    d = _LegworkDispatch(pool=None, qdrant=None, embed_query=lambda q: ([0.0], {}))  # type: ignore[arg-type]
    d.facts["AAPL"] = [
        _fact("rev", "FY2024"),
        _fact("rev", "FY2024"),  # duplicate
        _fact("rev", "FY2023"),
        _fact("rev", "FY2022"),
        _fact("rev", "FY2021"),
        _fact("rev", "FY2020"),  # older than the 4-year window
    ]
    quant, qual = d.evidence(_task())
    assert qual == {}
    assert [f.period for f in quant["AAPL"]] == ["FY2024", "FY2023", "FY2022", "FY2021"]
    assert quant["MSFT"] == []  # keyed by every task ticker


def test_legwork_evidence_dedups_and_caps_qual() -> None:
    d = _LegworkDispatch(pool=None, qdrant=None, embed_query=lambda q: ([0.0], {}))  # type: ignore[arg-type]
    d.hits["AAPL"] = [_hit(f"c{i}") for i in range(10)] + [_hit("c0")]  # 10 unique + 1 dup
    quant, qual = d.evidence(_task(mode="semantic"))
    assert quant == {}
    assert len(qual["AAPL"]) == 8  # deduped then capped
    assert len({h.chunk_id for h in qual["AAPL"]}) == 8


def _sentinel() -> AxisResult:
    return AxisResult(
        axis="profitability",
        mode="structured",
        per_company={},
        comparison="SENTINEL",
        citations=[],
        grounding="ok",
    )


def test_agentic_falls_back_when_legwork_gathers_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = _sentinel()
    monkeypatch.setattr(az, "analyze_axis", lambda task, **kw: sentinel)
    out = analyze_axis_agentic(
        _task(),
        legwork_client=_end_turn_client(),
        sonnet_client=_FakeSonnet("{}"),
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        embed_query=lambda q: ([0.0], {}),
    )
    assert out is sentinel


def test_agentic_falls_back_on_legwork_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = _sentinel()
    monkeypatch.setattr(az, "analyze_axis", lambda task, **kw: sentinel)

    class _Raise:
        backend = "anthropic"
        model = "claude-haiku-4-5"

        def chat(self, **kwargs: Any) -> Any:
            raise RuntimeError("503")

    out = analyze_axis_agentic(
        _task(),
        legwork_client=_Raise(),
        sonnet_client=_FakeSonnet("{}"),
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        embed_query=lambda q: ([0.0], {}),
    )
    assert out is sentinel


def test_agentic_falls_back_on_non_anthropic_legwork_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # run_agent_loop speaks the Anthropic wire format only. An OpenAI-protocol
    # (vLLM) legwork client must fail fast into the single-shot fallback with a
    # surfaced reason, not a swallowed TypeError.
    sentinel = _sentinel()
    monkeypatch.setattr(az, "analyze_axis", lambda task, **kw: sentinel)

    class _Vllm:
        backend = "vllm"
        model = "Qwen/Qwen2.5-7B-Instruct-AWQ"

        def chat(self, **kwargs: Any) -> Any:
            raise AssertionError("must not be called: backend guard fails fast")

    out = analyze_axis_agentic(
        _task(),
        legwork_client=_Vllm(),
        sonnet_client=_FakeSonnet("{}"),
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        embed_query=lambda q: ([0.0], {}),
    )
    assert out is sentinel


def test_write_axis_result_cites_only_gathered_evidence() -> None:
    # The agentic write phase: citations are code-built from the gathered facts,
    # so a ticker the legwork did not gather for cannot acquire a citation.
    facts = [_fact("profitability.revenue", "FY2024")]
    fake = _FakeSonnet(
        '{"comparison":"AAPL led [AAPL:Q0].","per_company":{"AAPL":"strong","MSFT":"thin"},'
        '"grounding":"ok","reason_if_not_ok":""}'
    )
    out = _write_axis_result(
        _task(),
        quant_evidence={"AAPL": facts, "MSFT": []},
        qual_evidence={},
        sonnet_client=fake,
        deadline=time.monotonic() + 10,
    )
    assert isinstance(out, AxisResult)
    assert {c.concept for c in out.citations if c.kind == "quant"} == {"profitability.revenue"}
    assert all(c.ticker == "AAPL" for c in out.citations)  # MSFT gathered nothing
