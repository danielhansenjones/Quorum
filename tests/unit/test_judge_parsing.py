from __future__ import annotations

from quorum.eval.judges import _coerce_score, _safe_json, score_report_quality
from quorum.eval.runner import CaseResult, _aggregate_judging


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _FakeJudge:
    backend = "anthropic"
    model = "fake"

    def __init__(self, text: str) -> None:
        self._text = text

    def chat(self, **_: object) -> _Resp:
        return _Resp(self._text)


def test_coerce_score_clamps_and_rejects() -> None:
    assert _coerce_score(5) == 5
    assert _coerce_score("4") == 4
    assert _coerce_score(7) == 5
    assert _coerce_score(0) == 1
    assert _coerce_score("nope") is None
    assert _coerce_score(None) is None


def test_safe_json_salvages_prose_wrapped() -> None:
    assert _safe_json('Here you go: {"score": 4, "reason": "ok"} thanks') == {
        "score": 4,
        "reason": "ok",
    }


def test_safe_json_truncated_returns_empty() -> None:
    # No closing brace: nothing to salvage. Caller must treat this as a failure,
    # not a zero-score rubric.
    assert _safe_json('{"clarity": 5, "comparison_quality": 5, "notes": "the report is') == {}


def test_quality_parses_well_formed() -> None:
    judge = _FakeJudge(
        '{"clarity": 5, "comparison_quality": 4, "evidence_coverage": 4, '
        '"honesty_on_insufficient_data": 5, "notes": "solid"}'
    )
    out = score_report_quality("a report", judge_client=judge)
    assert out["clarity"] == 5
    assert out["comparison_quality"] == 4
    assert out["notes"] == "solid"
    assert "judge_error" not in out


def test_quality_truncated_surfaces_error_not_all_ones() -> None:
    # The judged-full-v1 bug: truncated JSON used to become 1/1/1/1 silently.
    judge = _FakeJudge('{"clarity": 5, "comparison_quality": 5, "notes": "the report is excell')
    out = score_report_quality("a report", judge_client=judge)
    assert out.get("judge_error") == "unparseable_quality_response"
    assert not any(isinstance(v, int) for v in out.values())


def _case(case_id: str, quality: dict[str, object], status: str = "ok") -> CaseResult:
    return CaseResult(
        case_id=case_id,
        request_id=f"req-{case_id}",
        final_status=status,
        report="r",
        citations=[],
        elapsed_s=1.0,
        extra={"scores": {"faithfulness": {"n": 0, "mean_score": None}, "quality": quality}},
    )


def _rubric(score: int) -> dict[str, object]:
    return {
        "clarity": score,
        "comparison_quality": score,
        "evidence_coverage": score,
        "honesty_on_insufficient_data": score,
        "notes": "x",
    }


def test_aggregate_excludes_quality_judge_failures() -> None:
    good = _case("good", _rubric(4))
    failed = _case("failed", {"judge_error": "unparseable_quality_response", "notes": ""})
    agg = _aggregate_judging([good, failed])
    assert agg["quality_mean"] == 4.0
    assert agg["quality_judge_failures"] == 1


def test_aggregate_quality_excludes_refusals() -> None:
    # A refused case's message must not count toward report quality.
    answered = _case("answered", _rubric(5))
    refused = _case("refused_case", _rubric(1), status="refused")
    agg = _aggregate_judging([answered, refused])
    assert agg["quality_mean"] == 5.0
