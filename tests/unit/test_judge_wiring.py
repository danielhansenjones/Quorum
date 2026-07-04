from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from diskcache import Cache

from quorum.eval.judges import (
    ClaimVerdict,
    aggregate_faithfulness,
    reconstruct_citation,
    score_case,
    score_incorporation,
    score_report_quality,
    verify_qual_citation,
)
from quorum.state.citation import QualCitation, QuantCitation
from quorum.state.critique import FlaggedClaim


def _v(score: int, faithful: bool) -> ClaimVerdict:
    return ClaimVerdict(claim="c", faithful=faithful, score=score, reason="", judge="deterministic")


def test_aggregate_empty() -> None:
    out = aggregate_faithfulness([])
    assert out == {"n": 0, "mean_score": None, "faithful_fraction": None, "judge_failures": 0}


def test_aggregate_mixed() -> None:
    out = aggregate_faithfulness([_v(5, True), _v(5, True), _v(1, False), _v(2, False)])
    assert out["n"] == 4
    assert out["mean_score"] == (5 + 5 + 1 + 2) / 4
    assert out["faithful_fraction"] == 0.5
    assert out["judge_failures"] == 0


def test_aggregate_excludes_and_counts_error_verdicts() -> None:
    # An unparseable judge response must not fold into the mean as a
    # contradicted claim; it is excluded-and-counted, mirroring the quality
    # rubric's judge_error handling.
    err = ClaimVerdict(
        claim="c", faithful=False, score=1, reason="judge response unparseable", judge="error"
    )
    out = aggregate_faithfulness([_v(5, True), err])
    assert out["n"] == 1
    assert out["mean_score"] == 5.0
    assert out["faithful_fraction"] == 1.0
    assert out["judge_failures"] == 1


def test_reconstruct_quant() -> None:
    d = {
        "kind": "quant",
        "claim": "c",
        "ticker": "AAPL",
        "accession": "0000320193-24-000123",
        "concept": "us-gaap:Revenues",
        "value": "391035000000",
        "period": "FY2024",
        "unit": "USD",
    }
    c = reconstruct_citation(d)
    assert isinstance(c, QuantCitation)
    assert c.ticker == "AAPL"


def test_reconstruct_qual() -> None:
    d = {
        "kind": "qual",
        "claim": "c",
        "ticker": "JNJ",
        "accession": "a",
        "section": "item_1a_risk_factors",
        "chunk_id": "x",
        "quote": "q",
    }
    c = reconstruct_citation(d)
    assert isinstance(c, QualCitation)


def test_reconstruct_unknown_kind_returns_none() -> None:
    assert reconstruct_citation({"kind": "bogus"}) is None
    assert reconstruct_citation({}) is None


def test_reconstruct_malformed_returns_none() -> None:
    # Missing required fields must not raise.
    assert reconstruct_citation({"kind": "quant", "ticker": "AAPL"}) is None


# ---- 12d: critique-incorporation scorer ----


def _fc(claim: str, axis: str = "profitability", flag: str = "unsupported") -> FlaggedClaim:
    return FlaggedClaim(source_axis=axis, claim=claim, flag=flag, reason="r")


def test_incorporation_verbatim_repeat_scores_fail() -> None:
    claim = "Revenue grew 99% year over year"
    report = f"### Profitability\n{claim}. Both firms expanded."
    out = score_incorporation(report, [_fc(claim)])
    assert out["n"] == 1
    assert out["incorporated"] == 0
    assert out["rate"] == 0.0
    assert out["claims"][0]["incorporated"] is False


def test_incorporation_softened_scores_pass() -> None:
    claim = "Revenue grew 99% year over year"
    softened = (
        "### Profitability\nRevenue rose, though the exact rate is not separable from evidence."
    )
    out = score_incorporation(softened, [_fc(claim)])
    assert out["incorporated"] == 1
    assert out["rate"] == 1.0
    assert out["claims"][0]["incorporated"] is True


def test_incorporation_rate_aggregates_across_claims() -> None:
    kept = "AAPL margin is 46%"
    dropped = "MSFT tripled its debt"
    report = f"Analysis: {kept}. Nothing else."
    out = score_incorporation(report, [_fc(kept), _fc(dropped)])
    assert out["n"] == 2
    assert out["incorporated"] == 1  # kept verbatim fails; dropped passes
    assert out["rate"] == 0.5


def test_incorporation_empty_claims_rate_none() -> None:
    out = score_incorporation("any report", [])
    assert out["n"] == 0
    assert out["rate"] is None


class _Judge:
    backend = "anthropic"
    model = "claude-sonnet-4-6"

    def chat(self, **kwargs: Any) -> Any:
        class _Block:
            text = (
                '{"clarity":5,"comparison_quality":5,"evidence_coverage":5,'
                '"honesty_on_insufficient_data":5,"notes":"ok"}'
            )

        class _Resp:
            content = [_Block()]

        return _Resp()


def test_score_case_attaches_incorporation_when_flagged_claims_passed() -> None:
    out = score_case(
        report="Some report with AAPL margin is 46%.",
        citations=[],
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        judge_client=_Judge(),
        flagged_claims=[_fc("AAPL margin is 46%")],
    )
    assert out["incorporation"]["n"] == 1
    assert out["incorporation"]["incorporated"] == 0


def test_score_case_persists_per_claim_verdicts(monkeypatch: Any) -> None:
    # Verdicts ride in the artifact next to the aggregate so error verdicts are
    # auditable per case, not invisible behind a mean.
    monkeypatch.setattr("quorum.eval.judges.verify_quant_citation", lambda pool, c: _v(5, True))
    out = score_case(
        report="Some report.",
        citations=[
            {
                "kind": "quant",
                "claim": "c",
                "ticker": "AAPL",
                "accession": "a",
                "concept": "us-gaap:Revenues",
                "value": "1",
                "period": "FY2024",
                "unit": "USD",
            }
        ],
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        judge_client=_Judge(),
    )
    claims = out["faithfulness"]["claims"]
    assert len(claims) == 1
    assert claims[0]["judge"] == "deterministic"
    assert claims[0]["score"] == 5
    assert out["faithfulness"]["judge_failures"] == 0


def test_score_case_omits_incorporation_without_flagged_claims() -> None:
    # Backward compat: no flagged_claims -> no block (empty report -> no judge call).
    out = score_case(
        report="",
        citations=[],
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        judge_client=_Judge(),
    )
    assert "incorporation" not in out


# ---- 13: judge calls ride the shared LLM disk cache ----


# Module-level so diskcache can pickle the cached response.
class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _CountingJudge:
    backend = "anthropic"
    model = "claude-sonnet-4-6"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def chat(self, **kwargs: Any) -> Any:
        self.calls += 1
        return _Resp(self.text)


def test_quality_judge_cached_second_call_is_free(tmp_path: Any) -> None:
    judge = _CountingJudge(
        '{"clarity":5,"comparison_quality":4,"evidence_coverage":5,'
        '"honesty_on_insufficient_data":5,"notes":"ok"}'
    )
    cache = Cache(str(tmp_path))
    first = score_report_quality("Report A", judge_client=judge, llm_cache=cache)
    second = score_report_quality("Report A", judge_client=judge, llm_cache=cache)
    assert judge.calls == 1
    assert first == second
    # A different report is a different key, not a stale hit.
    score_report_quality("Report B", judge_client=judge, llm_cache=cache)
    assert judge.calls == 2


def test_quality_judge_rubric_change_misses(tmp_path: Any, monkeypatch: Any) -> None:
    # An edited rubric must miss without a prompt_version bump; a stale hit
    # would silently score reports under the old rubric across runs.
    judge = _CountingJudge(
        '{"clarity":5,"comparison_quality":4,"evidence_coverage":5,'
        '"honesty_on_insufficient_data":5,"notes":"ok"}'
    )
    cache = Cache(str(tmp_path))
    score_report_quality("Report A", judge_client=judge, llm_cache=cache)
    monkeypatch.setattr("quorum.eval.judges._QUALITY_SYSTEM", "edited rubric text")
    score_report_quality("Report A", judge_client=judge, llm_cache=cache)
    assert judge.calls == 2


def test_qual_citation_judge_cached_second_call_is_free(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "quorum.eval.judges.get_filing_section",
        lambda qdrant, **kwargs: SimpleNamespace(text="Supply chain risks increased."),
    )
    citation = QualCitation(
        kind="qual",
        claim="Risks increased",
        ticker="JNJ",
        accession="a",
        section="item_1a_risk_factors",
        chunk_id="x",
        quote="",
    )
    judge = _CountingJudge('{"score": 5, "reason": "supported"}')
    cache = Cache(str(tmp_path))
    first = verify_qual_citation(None, citation, judge_client=judge, llm_cache=cache)  # type: ignore[arg-type]
    second = verify_qual_citation(None, citation, judge_client=judge, llm_cache=cache)  # type: ignore[arg-type]
    assert judge.calls == 1
    assert first == second
    assert first.score == 5
