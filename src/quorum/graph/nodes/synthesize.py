from __future__ import annotations

from typing import Any

from diskcache import Cache

from quorum.models.cached_chat import chat_maybe_cached
from quorum.models.router import ChatClient
from quorum.state.axis import AxisResult
from quorum.state.citation import Citation
from quorum.state.critique import Critique, Rebuttal
from quorum.trace.cost import llm_trace_fields
from quorum.trace.writer import TraceCtx

_SYNTHESIZER_SYSTEM = (
    "You are Quorum's synthesizer. Given per-axis comparisons and an optional "
    "critic report, write the final multi-axis report.\n\n"
    "Rules:\n"
    "- One markdown section per axis present in axis_results.\n"
    "- For each grounded axis: write the cross-company comparison paragraph first, then one "
    "labeled line per company in the form '**TICKER:** <assessment>' using the per-company reads provided.\n"
    "- An axis may carry a '_Caveat: ...' note (partial data, e.g. a metric missing for one company). "
    "Render that axis in full and keep the caveat as an italic note; do NOT drop it to '*Insufficient data*'.\n"
    "- For each axis marked '*Insufficient data*' in the input, write '*Insufficient data*' as the body.\n"
    "- If a critic flagged a claim, drop, soften, or counter-cite it. Never repeat "
    "  a flagged claim verbatim.\n"
    "- Preserve [TICKER:CITE_ID] markers from the axis comparisons; do not invent new ones.\n"
    "- Conclude with a short cross-axis takeaway only if at least two axes are grounded."
)

_INSUFFICIENT_MARKER = "*Insufficient data*"


def _format_axis_result(r: AxisResult) -> str:
    head = f"### {r.axis.replace('_', ' ').title()}\n"
    renderable = bool(r.comparison.strip()) or bool(r.citations)
    if r.grounding == "insufficient" or not renderable:
        return head + _INSUFFICIENT_MARKER + (f" ({r.error_reason})" if r.error_reason else "")
    body = r.comparison.strip() or _INSUFFICIENT_MARKER
    per_company = [
        f"- {f.ticker}: {f.assessment.strip()}"
        for f in r.per_company.values()
        if f.assessment.strip()
    ]
    if per_company:
        body += "\n\nPer company:\n" + "\n".join(per_company)
    # A weak axis is rendered with its evidence but flagged: the data is partial
    # (e.g. a sub-metric missing for one company), not absent.
    if r.grounding == "weak" and r.error_reason:
        body += f"\n\n_Caveat: {r.error_reason}_"
    return head + body


def _format_critique(c: Critique | None) -> str:
    if c is None:
        return (
            "Critique: unavailable (critic was bypassed, timed out, or failed). "
            "No claims have been flagged."
        )
    if not c.flagged_claims:
        return f"Critique status={c.status}. No flagged claims."
    flags = "\n".join(
        f"- [{f.source_axis}] {f.flag}: {f.claim}\n  Reason: {f.reason}" for f in c.flagged_claims
    )
    return f"Critique status={c.status}. Flagged claims:\n{flags}"


def _format_rebuttals(rebuttals: list[Rebuttal]) -> str:
    lines = []
    for r in rebuttals:
        cite = f" (defending cite: {r.citation.ticker})" if r.citation is not None else ""
        lines.append(f"- [{r.source_axis}] {r.disposition}: {r.claim}\n  Analyst: {r.reason}{cite}")
    return (
        "Analyst rebuttals to the flagged claims (adjudicate each):\n"
        + "\n".join(lines)
        + "\n\nFor a RETRACTED claim, drop it. For a DEFENDED or REVISED claim, keep the "
        "corrected statement and its citation."
    )


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def _apply_rebuttals(report: str, rebuttals: list[Rebuttal]) -> tuple[str, list[Citation]]:
    # Deterministic adjudication on top of the LLM's prose: a retracted claim is
    # stripped verbatim (the LLM is told to drop it, this enforces it); a
    # defended/revised claim contributes its citation to the report.
    retracted = [
        _normalize(r.claim) for r in rebuttals if r.disposition == "retracted" and r.claim.strip()
    ]
    extra = [
        r.citation
        for r in rebuttals
        if r.disposition in ("defended", "revised") and r.citation is not None
    ]
    if not retracted:
        return report, extra
    kept = [ln for ln in report.splitlines() if not any(rc in _normalize(ln) for rc in retracted)]
    return "\n".join(kept), extra


def _strip_uncited(report: str) -> str:
    # Phase 6f structural check: drop any line that makes a numeric claim with
    # no [TICKER:ID] marker. v1 implementation: simple regex-based scan kept
    # conservative; production-grade would use a proper claim parser.
    import re

    out_lines: list[str] = []
    for line in report.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(_INSUFFICIENT_MARKER):
            out_lines.append(line)
            continue
        # If a line contains digits or currency markers, require a [...:...] cite.
        has_number = bool(re.search(r"\d", s)) or "$" in s
        has_cite = bool(re.search(r"\[[A-Z]+:[A-Za-z0-9_:-]+\]", s))
        if has_number and not has_cite:
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _extract_text(resp: Any) -> str:
    blocks = getattr(resp, "content", None) or []
    for b in blocks:
        t = getattr(b, "text", None)
        if t:
            return str(t)
    return ""


def synthesize(
    *,
    axis_results: list[AxisResult],
    critique: Critique | None,
    sonnet_client: ChatClient,
    question: str,
    rebuttals: list[Rebuttal] | None = None,
    llm_cache: Cache | None = None,
    prompt_version: str = "synthesizer-v1",
    trace_ctx: TraceCtx | None = None,
) -> dict[str, Any]:
    # Phase 6f. Sonnet, cacheable system prompt. Strip uncited numeric claims
    # before emitting. Insufficient axes get the explicit marker, not a fabricated
    # number. Phase 13a: adjudicate analyst rebuttals to the critic's flags.
    axis_blocks = "\n\n".join(_format_axis_result(r) for r in axis_results)
    critique_text = _format_critique(critique)
    # Guarded so an empty rebuttal list leaves the user message (and cache key)
    # identical to the no-rebuttal arm.
    rebuttal_text = f"{_format_rebuttals(rebuttals)}\n\n" if rebuttals else ""
    user = (
        f"Question:\n{question}\n\n"
        f"Per-axis comparisons:\n{axis_blocks}\n\n"
        f"{critique_text}\n\n"
        f"{rebuttal_text}"
        "Write the final report now."
    )
    try:
        resp = chat_maybe_cached(
            sonnet_client,
            llm_cache,
            prompt_version=prompt_version,
            system=[
                {
                    "type": "text",
                    "text": _SYNTHESIZER_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=2048,
        )
        if trace_ctx is not None:
            trace_ctx.event("llm:synthesizer", **llm_trace_fields(sonnet_client.model, resp))
        text = _extract_text(resp)
    except Exception as e:  # noqa: BLE001
        return {
            "report": (
                f"Synthesis failed: {type(e).__name__}.\n\nPartial axis results:\n" + axis_blocks
            ),
            "status": "partial",
            "report_citations": _collect_citations(axis_results),
        }

    cleaned = _strip_uncited(text)
    cleaned, rebuttal_citations = _apply_rebuttals(cleaned, rebuttals or [])
    status = "ok"
    # An axis flagged weak/insufficient upstream, OR a section the synthesizer
    # rendered as insufficient (e.g. a requested sub-metric with no isolable
    # data), both mean the answer is incomplete. Either makes the run partial.
    if (
        any(r.grounding in ("weak", "insufficient") for r in axis_results)
        or _INSUFFICIENT_MARKER in cleaned
    ):
        status = "partial"

    return {
        "report": cleaned,
        "status": status,
        "report_citations": _collect_citations(axis_results) + rebuttal_citations,
    }


def _collect_citations(axis_results: list[AxisResult]) -> list[Citation]:
    out: list[Citation] = []
    for r in axis_results:
        out.extend(r.citations)
    return out
