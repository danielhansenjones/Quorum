from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from diskcache import Cache
from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient

from quorum.models.cached_chat import cached_chat
from quorum.models.router import ChatClient
from quorum.state.citation import QualCitation, QuantCitation
from quorum.state.critique import FlaggedClaim
from quorum.tools.concept_resolver import get_financial_concept
from quorum.tools.filing_section import FilingSectionNotFound, get_filing_section


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    claim: str
    faithful: bool
    score: int  # 1-5 Likert
    reason: str
    judge: str  # "deterministic" | "local" | "sonnet"


# ---- Faithfulness: quant deterministic ----


def verify_quant_citation(
    pool: ConnectionPool,
    citation: QuantCitation,
    *,
    tolerance_relative: float = 1e-6,
) -> ClaimVerdict:
    # Phase 10d: quantitative citations are deterministic. Pull the underlying
    # (cik, concept, period) from Postgres and compare value, unit, period.
    facts = get_financial_concept(
        pool, ticker=citation.ticker, key=citation.concept, periods=[citation.period]
    )
    if not facts:
        return ClaimVerdict(
            claim=citation.claim,
            faithful=False,
            score=1,
            reason="No matching fact in Postgres for cited concept and period.",
            judge="deterministic",
        )
    fact = facts[0]
    cited_value = _parse_numeric(citation.value)
    if cited_value is None or fact.value == 0:
        # Fall back to string equality if the citation value isn't a clean number.
        ok = str(fact.value) == citation.value and fact.unit == citation.unit
        return ClaimVerdict(
            claim=citation.claim,
            faithful=ok,
            score=5 if ok else 2,
            reason="String comparison" if ok else "Citation value does not parse as numeric.",
            judge="deterministic",
        )
    rel_err = abs(fact.value - cited_value) / abs(fact.value)
    ok_value = rel_err <= tolerance_relative
    ok_unit = fact.unit == citation.unit
    score = 5 if (ok_value and ok_unit) else (2 if ok_value else 1)
    return ClaimVerdict(
        claim=citation.claim,
        faithful=ok_value and ok_unit,
        score=score,
        reason=f"rel_err={rel_err:.2e}, unit_match={ok_unit}",
        judge="deterministic",
    )


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_numeric(s: str) -> float | None:
    m = _NUMERIC_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


# ---- Faithfulness: qual + quality (LLM-based) ----


_FAITHFULNESS_SYSTEM = (
    "You are an LLM-as-judge for Quorum reports. Given a claim and a supporting "
    "passage, score how well the passage supports the claim on a 1-5 Likert scale.\n\n"
    "1: Contradicted by the passage.\n"
    "2: Unsupported by the passage.\n"
    "3: Partially supported; some detail matches.\n"
    "4: Supported; small nuance missing.\n"
    "5: Fully supported.\n\n"
    'Respond with ONLY a JSON object: {"score": int, "reason": str}.'
)


def verify_qual_citation(
    qdrant: QdrantClient,
    citation: QualCitation,
    *,
    judge_client: ChatClient,
    llm_cache: Cache | None = None,
) -> ClaimVerdict:
    try:
        section = get_filing_section(
            qdrant, ticker=citation.ticker, accession=citation.accession, section=citation.section
        )
    except FilingSectionNotFound:
        return ClaimVerdict(
            claim=citation.claim,
            faithful=False,
            score=1,
            reason="Cited section not found in corpus.",
            judge="deterministic",
        )
    if citation.quote and citation.quote not in section.text:
        return ClaimVerdict(
            claim=citation.claim,
            faithful=False,
            score=1,
            reason="Quoted span not present in the cited section.",
            judge="deterministic",
        )
    user = f"Claim: {citation.claim}\n\nSection excerpt (first 4000 chars):\n{section.text[:4000]}"
    resp = _judge_chat(
        judge_client,
        system=_FAITHFULNESS_SYSTEM,
        user=user,
        max_tokens=384,
        cache=llm_cache,
        prompt_version="judge-faithfulness-v1",
    )
    text = _extract_text(resp, judge_client.backend)
    parsed = _safe_json(text)
    raw_score = _coerce_score(parsed.get("score"))
    if raw_score is None:
        # Unparseable judge output must not masquerade as a contradicted claim;
        # surface it so the aggregate can be read with eyes open.
        return ClaimVerdict(
            claim=citation.claim,
            faithful=False,
            score=1,
            reason="judge response unparseable",
            judge="error",
        )
    reason = str(parsed.get("reason", ""))
    return ClaimVerdict(
        claim=citation.claim,
        faithful=raw_score >= 4,
        score=raw_score,
        reason=reason,
        judge="local" if judge_client.backend == "vllm" else "sonnet",
    )


_QUALITY_DIMENSIONS = (
    "clarity",
    "comparison_quality",
    "evidence_coverage",
    "honesty_on_insufficient_data",
)


_QUALITY_SYSTEM = (
    "You are an LLM-as-judge for the Quorum report's writing quality. Score each "
    "dimension 1-5. Respond with ONLY a JSON object:\n"
    '{ "clarity": int, "comparison_quality": int, "evidence_coverage": int, '
    '"honesty_on_insufficient_data": int, "notes": str }'
)


def score_report_quality(
    report: str, *, judge_client: ChatClient, llm_cache: Cache | None = None
) -> dict[str, Any]:
    # max_tokens must clear four integer scores plus a free-text `notes`; at 256
    # the notes field truncated mid-string on long reports, json.loads failed,
    # and every dimension silently defaulted to 1.
    resp = _judge_chat(
        judge_client,
        system=_QUALITY_SYSTEM,
        user=f"REPORT:\n{report[:6000]}",
        max_tokens=700,
        cache=llm_cache,
        prompt_version="judge-quality-v1",
    )
    parsed = _safe_json(_extract_text(resp, judge_client.backend))
    scores = {d: s for d in _QUALITY_DIMENSIONS if (s := _coerce_score(parsed.get(d))) is not None}
    if not scores:
        # No usable rubric (truncated / unparseable). Surface it rather than
        # scoring a good report 1/1/1/1; the aggregator excludes and counts it.
        return {"judge_error": "unparseable_quality_response", "notes": ""}
    return scores | {"notes": str(parsed.get("notes", ""))}


def reconstruct_citation(d: dict[str, Any]) -> QuantCitation | QualCitation | None:
    # CaseResult stores citations as model_dump dicts; rebuild the typed model
    # via the `kind` discriminator so the judges can re-verify them.
    kind = d.get("kind")
    try:
        if kind == "quant":
            return QuantCitation(**d)
        if kind == "qual":
            return QualCitation(**d)
    except Exception:  # noqa: BLE001
        return None
    return None


def aggregate_faithfulness(verdicts: list[ClaimVerdict]) -> dict[str, Any]:
    # Pure aggregation over per-citation verdicts (no IO; unit-tested directly).
    n = len(verdicts)
    if n == 0:
        return {"n": 0, "mean_score": None, "faithful_fraction": None}
    return {
        "n": n,
        "mean_score": sum(v.score for v in verdicts) / n,
        "faithful_fraction": sum(1 for v in verdicts if v.faithful) / n,
    }


def _normalize_text(s: str) -> str:
    # Collapse whitespace and lowercase so a "verbatim" repeat is detected past
    # markdown reflow / capitalization, without crediting a real paraphrase.
    return " ".join(s.split()).lower()


def score_incorporation(report: str, flagged_claims: list[FlaggedClaim]) -> dict[str, Any]:
    # Phase 12d. A critic flag is "incorporated" if synthesize dropped, softened,
    # or counter-cited it - all of which change the claim text. v1 proxy: the
    # claim is NOT present verbatim in the final report. Deterministic, no judge.
    norm_report = _normalize_text(report)
    claims: list[dict[str, Any]] = []
    for fc in flagged_claims:
        claim_norm = _normalize_text(fc.claim)
        repeated = bool(claim_norm) and claim_norm in norm_report
        claims.append(
            {
                "source_axis": fc.source_axis,
                "claim": fc.claim,
                "flag": fc.flag,
                "incorporated": not repeated,
            }
        )
    n = len(claims)
    incorporated = sum(1 for c in claims if c["incorporated"])
    return {
        "n": n,
        "incorporated": incorporated,
        "rate": (incorporated / n) if n else None,
        "claims": claims,
    }


def score_case(
    *,
    report: str,
    citations: list[dict[str, Any]],
    pool: ConnectionPool,
    qdrant: QdrantClient,
    judge_client: ChatClient,
    flagged_claims: list[FlaggedClaim] | None = None,
    llm_cache: Cache | None = None,
) -> dict[str, Any]:
    # Faithfulness: deterministic for quant citations, LLM-judged for qual.
    # Quality: one LLM rubric pass over the whole report.
    verdicts: list[ClaimVerdict] = []
    for d in citations:
        c = reconstruct_citation(d)
        if c is None:
            continue
        if isinstance(c, QuantCitation):
            verdicts.append(verify_quant_citation(pool, c))
        else:
            verdicts.append(
                verify_qual_citation(qdrant, c, judge_client=judge_client, llm_cache=llm_cache)
            )
    quality = (
        score_report_quality(report, judge_client=judge_client, llm_cache=llm_cache)
        if report.strip()
        else None
    )
    out: dict[str, Any] = {"faithfulness": aggregate_faithfulness(verdicts), "quality": quality}
    # Only when a critique was captured (flagged_claims may be an empty list:
    # a critique that flagged nothing still scores n=0, distinct from None).
    if flagged_claims is not None:
        out["incorporation"] = score_incorporation(report, flagged_claims)
    return out


def _judge_chat(
    judge_client: ChatClient,
    *,
    system: str,
    user: str,
    max_tokens: int,
    cache: Cache | None = None,
    prompt_version: str = "",
) -> Any:
    # Anthropic takes `system` as a top-level kwarg; OpenAI/vLLM want it as the
    # first message. The judges must work with both for the 10c local-vs-canonical
    # correlation study.
    if judge_client.backend == "anthropic":
        # `system` rides in chat_kwargs and is hashed into the cache key, so an
        # edited rubric misses on its own - same contract as chat_maybe_cached
        # in the graph nodes. prompt_version stays as an explicit lever.
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        chat_kwargs: dict[str, Any] | None = {"system": system}
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        chat_kwargs = None
    if cache is None:
        return judge_client.chat(
            messages=messages, max_tokens=max_tokens, temperature=0.0, **(chat_kwargs or {})
        )
    return cached_chat(
        judge_client,
        cache,
        prompt_version=prompt_version,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        chat_kwargs=chat_kwargs,
    )


def _extract_text(resp: Any, backend: str) -> str:
    if backend == "anthropic":
        blocks = getattr(resp, "content", None) or []
        for b in blocks:
            t = getattr(b, "text", None)
            if t:
                return str(t)
        return ""
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    return str(getattr(getattr(choices[0], "message", None), "content", "") or "")


def _coerce_score(v: Any) -> int | None:
    try:
        return max(1, min(5, int(v)))
    except (TypeError, ValueError):
        return None


def _safe_json(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
    s = s.strip()
    try:
        return dict(json.loads(s))
    except json.JSONDecodeError:
        pass
    # Salvage a single top-level object embedded in surrounding prose. True
    # truncation (no closing brace) still falls through to {} -> caller decides.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            return dict(json.loads(s[start : end + 1]))
        except json.JSONDecodeError:
            return {}
    return {}
