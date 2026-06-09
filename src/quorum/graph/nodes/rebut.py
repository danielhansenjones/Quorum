from __future__ import annotations

import json
import re
from typing import Any

from diskcache import Cache

from quorum.graph.nodes.analyze_axis import _extract_first_json_object, _extract_text
from quorum.models.cached_chat import chat_maybe_cached
from quorum.models.router import ChatClient
from quorum.state.axis import AxisResult
from quorum.state.citation import Citation
from quorum.state.critique import FlaggedClaim, Rebuttal
from quorum.trace.cost import llm_trace_fields
from quorum.trace.writer import TraceCtx

_REBUT_SYSTEM = (
    "You are a Quorum axis analyst responding to a critic that flagged claims you made. "
    "For each flagged claim decide: DEFEND it (point to a supporting evidence row), RETRACT "
    "it (you cannot support it), or REVISE it (give a corrected, supportable version). Emit "
    "ONE JSON object:\n"
    "{\n"
    '  "rebuttals": [\n'
    '    {"claim": str,            # the flagged claim, verbatim\n'
    '     "disposition": "defended" | "retracted" | "revised",\n'
    '     "reason": str,           # 1-2 sentences\n'
    '     "cite_ref": str | null}  # id of a supporting evidence row (e.g. "C2") when defending/revising\n'
    "  ]\n"
    "}\n"
    "Rules:\n"
    "- Only cite an evidence id that appears under AVAILABLE EVIDENCE. Never invent one.\n"
    "- Retract rather than defend with weak or absent evidence. Honesty over winning.\n"
    "- Address every flagged claim exactly once."
)

_DISPOSITIONS: set[str] = {"defended", "retracted", "revised"}
_CITE_REF = re.compile(r"\d+")


def _citation_brief(i: int, c: Citation) -> str:
    if c.kind == "quant":
        return f"C{i}: quant {c.ticker} {c.concept} {c.period} = {c.value} {c.unit}"
    return f"C{i}: qual {c.ticker} {c.section} chunk={c.chunk_id}"


def _resolve_cite_ref(ref: Any, citations: list[Citation]) -> Citation | None:
    if not isinstance(ref, str | int):
        return None
    m = _CITE_REF.search(str(ref))
    if not m:
        return None
    idx = int(m.group(0))
    return citations[idx] if 0 <= idx < len(citations) else None


def _rebut_axis(
    axis: str,
    claims: list[FlaggedClaim],
    citations: list[Citation],
    *,
    sonnet_client: ChatClient,
    llm_cache: Cache | None,
    prompt_version: str,
    trace_ctx: TraceCtx | None,
) -> list[Rebuttal]:
    flagged_block = "\n".join(
        f"[F{i}] flag={c.flag} claim={c.claim}\n     critic reason: {c.reason}"
        for i, c in enumerate(claims)
    )
    evidence_block = (
        "\n".join(_citation_brief(i, c) for i, c in enumerate(citations)) or "(no evidence rows)"
    )
    user = (
        f"axis: {axis}\n\nFLAGGED CLAIMS\n--------------\n{flagged_block}\n\n"
        f"AVAILABLE EVIDENCE\n------------------\n{evidence_block}\n\n"
        "Emit the JSON object now."
    )
    resp = chat_maybe_cached(
        sonnet_client,
        llm_cache,
        prompt_version=prompt_version,
        system=[{"type": "text", "text": _REBUT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=1200,
    )
    if trace_ctx is not None:
        trace_ctx.event(
            "llm:rebut", **llm_trace_fields(sonnet_client.model, resp), input_shape={"axis": axis}
        )
    candidate = _extract_first_json_object(_extract_text(resp))
    if candidate is None:
        return []
    parsed = json.loads(candidate)
    out: list[Rebuttal] = []
    for item in parsed.get("rebuttals") or []:
        disposition = str(item.get("disposition", ""))
        if disposition not in _DISPOSITIONS:
            # Unaddressed / malformed: leave the claim to synthesize's critic-flag
            # fallback (drop/soften) rather than guessing a disposition.
            continue
        citation = (
            _resolve_cite_ref(item.get("cite_ref"), citations)
            if disposition in ("defended", "revised")
            else None
        )
        out.append(
            Rebuttal(
                source_axis=axis,
                claim=str(item.get("claim", "")),
                disposition=disposition,  # type: ignore[arg-type]
                reason=str(item.get("reason", "")),
                citation=citation,
            )
        )
    return out


def rebut(
    *,
    flagged_claims: list[FlaggedClaim],
    axis_results: list[AxisResult],
    remaining_steps: int,
    sonnet_client: ChatClient,
    llm_cache: Cache | None = None,
    prompt_version: str = "rebut-v1",
    trace_ctx: TraceCtx | None = None,
) -> dict[str, Any]:
    # Phase 13a. One rebuttal pass: re-invoke each flagged axis's analyst to
    # defend / retract / revise. Never raises (per-axis containment). Decrements
    # the shared budget so the critic <-> analyst loop is bounded.
    citations_by_axis: dict[str, list[Citation]] = {r.axis: list(r.citations) for r in axis_results}
    by_axis: dict[str, list[FlaggedClaim]] = {}
    for fc in flagged_claims:
        by_axis.setdefault(fc.source_axis, []).append(fc)

    rebuttals: list[Rebuttal] = []
    for axis, claims in by_axis.items():
        try:
            rebuttals.extend(
                _rebut_axis(
                    axis,
                    claims,
                    citations_by_axis.get(axis, []),
                    sonnet_client=sonnet_client,
                    llm_cache=llm_cache,
                    prompt_version=prompt_version,
                    trace_ctx=trace_ctx,
                )
            )
        except Exception:  # noqa: BLE001
            continue
    return {"rebuttals": rebuttals, "remaining_steps": max(0, remaining_steps - 1)}
