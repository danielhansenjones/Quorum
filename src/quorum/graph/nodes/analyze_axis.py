from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

from diskcache import Cache
from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient

from quorum.graph.agent_loop import run_agent_loop
from quorum.graph.axis_config import AXIS_CONCEPTS, AXIS_SEMANTIC
from quorum.graph.json_parse import extract_first_json_object

# The agentic legwork loop reuses the critic's retrieval tool schemas (the single
# source of truth for these three tools); only the dispatch differs. critic does
# not import this module, so the import is acyclic.
from quorum.graph.nodes.critic import _TOOL_DEFS as _LEGWORK_TOOL_DEFS
from quorum.models.cached_chat import chat_maybe_cached
from quorum.models.router import ChatClient
from quorum.state.axis import (
    AxisResult,
    AxisTask,
    CompanyAxisFinding,
    Grounding,
)
from quorum.state.citation import Citation, QualCitation, QuantCitation
from quorum.tools.concept_resolver import ResolvedFact, get_financial_concept
from quorum.tools.filing_section import get_filing_section
from quorum.tools.search import SearchHit, hybrid_search
from quorum.trace.cost import llm_trace_fields
from quorum.trace.writer import TraceCtx

DEFAULT_BRANCH_TIMEOUT_S = 60.0
MAX_RETRIES = 2

# Structured evidence ships annual-only, most recent N fiscal years. Quarterly
# slices and older years inflate the payload past the analyst's time budget
# without adding cross-company comparison signal (smoke 2026-05-28: a 4-concept
# 2-ticker profitability call shipped ~128 lines and blew the 60s branch budget).
MAX_FISCAL_YEARS = 4
_FY_PERIOD = re.compile(r"^FY(\d{4})$")

_ANALYST_SYSTEM = (
    "You are one of Quorum's axis analysts. You receive structured evidence "
    "(XBRL facts or filing passages) gathered by the planner for one axis "
    "across two or more companies. Your job: write a within-axis comparison "
    "grounded in the evidence and emit ONE JSON object matching this schema:\n"
    "{\n"
    '  "comparison": str,            # one paragraph comparing the companies, with [TICKER:CITE_ID] markers\n'
    '  "per_company": {              # ticker -> a 1-3 sentence read on how that company is doing on this axis\n'
    '    "<ticker>": "assessment prose with [TICKER:CITE_ID] markers"\n'
    "  },\n"
    '  "grounding": "ok" | "weak" | "insufficient",\n'
    '  "reason_if_not_ok": str       # short note if grounding != ok, else empty\n'
    "}\n"
    "Rules:\n"
    "- Cite every numerical claim or quoted span with a [TICKER:ID] marker matching an evidence row.\n"
    "- per_company values are PROSE, not tables. Do NOT dump the raw numbers back as a list; the figures "
    "are already recorded. Reference at most a couple of key figures inline and keep each assessment to 1-3 "
    "sentences.\n"
    "- Cover every ticker in per_company, even if the read is that evidence is thin for that company.\n"
    "- If evidence is thin or missing for one side, say so explicitly and grade `weak` or `insufficient`.\n"
    "- Do not invent numbers, periods, or units."
)


def _format_quant_evidence(ticker: str, facts: list[ResolvedFact]) -> str:
    if not facts:
        return f"[{ticker}] no XBRL facts retrieved\n"
    lines = []
    for i, f in enumerate(facts):
        lines.append(
            f"[{ticker}:Q{i}] {f.resolved_concept} {f.period} = {f.value} {f.unit} "
            f"(accession {f.accession})"
        )
    return "\n".join(lines) + "\n"


def _format_qual_evidence(ticker: str, hits: list[SearchHit]) -> str:
    if not hits:
        return f"[{ticker}] no chunks retrieved\n"
    lines = []
    for i, h in enumerate(hits):
        snippet = str(h.payload.get("text", ""))[:600]
        chunk_id = str(h.payload.get("chunk_id", h.chunk_id))
        section = str(h.payload.get("section", ""))
        lines.append(
            f"[{ticker}:S{i}] section={section} chunk_id={chunk_id} score={h.score:.3f}\n"
            f"  {snippet}"
        )
    return "\n".join(lines) + "\n"


def _trim_to_recent_fiscal_years(
    facts: list[ResolvedFact], *, max_years: int = MAX_FISCAL_YEARS
) -> list[ResolvedFact]:
    annual = [(m.group(1), f) for f in facts if (m := _FY_PERIOD.match(f.period))]
    if not annual:
        return []
    keep = set(sorted({yr for yr, _ in annual}, reverse=True)[:max_years])
    return [f for yr, f in annual if yr in keep]


def _gather_structured_evidence(
    task: AxisTask, *, pool: ConnectionPool
) -> dict[str, list[ResolvedFact]]:
    by_ticker: dict[str, list[ResolvedFact]] = {}
    keys = AXIS_CONCEPTS.get(task.axis, (task.query_or_concept,))
    for ticker in task.tickers:
        bucket: list[ResolvedFact] = []
        for key in keys:
            bucket.extend(get_financial_concept(pool, ticker=ticker, key=key))
        by_ticker[ticker] = _trim_to_recent_fiscal_years(bucket)
    return by_ticker


def _gather_semantic_evidence(
    task: AxisTask,
    *,
    qdrant: QdrantClient,
    embed_query: Callable[[str], tuple[list[float], dict[str, float]]],
    top_k: int = 5,
) -> dict[str, list[SearchHit]]:
    sem_cfg = AXIS_SEMANTIC.get(task.axis, {})
    sections = sem_cfg.get("sections") if isinstance(sem_cfg.get("sections"), list) else None
    query = task.query_or_concept
    dense, sparse = embed_query(query)
    by_ticker: dict[str, list[SearchHit]] = {}
    for ticker in task.tickers:
        by_ticker[ticker] = hybrid_search(
            qdrant,
            dense_vec=dense,
            sparse_weights=sparse,
            tickers=[ticker],
            sections=sections if isinstance(sections, list) else None,
            top_k=top_k,
        )
    return by_ticker


def _parse_analyst_output(raw: str) -> dict[str, Any]:
    # Smoke (2026-05-28) showed Sonnet frequently wraps the JSON object with
    # prose ("Here is the analysis:"), markdown fences, or trailing commentary.
    # Walk the response and pull the first balanced JSON object out instead of
    # assuming the response is exactly the JSON.
    candidate = extract_first_json_object(raw)
    if candidate is None:
        raise ValueError("no JSON object found in analyst output")
    return dict(json.loads(candidate))


def _extract_text(resp: Any) -> str:
    blocks = getattr(resp, "content", None) or []
    for b in blocks:
        t = getattr(b, "text", None)
        if t:
            return str(t)
    return ""


def _citations_for_quant(ticker: str, facts: list[ResolvedFact], claim: str) -> list[Citation]:
    return [
        QuantCitation(
            claim=claim,
            ticker=ticker,
            accession=f.accession,
            concept=f.resolved_concept,
            value=str(f.value),
            period=f.period,
            unit=f.unit,
        )
        for f in facts
    ]


def _citations_for_qual(ticker: str, hits: list[SearchHit], claim: str) -> list[Citation]:
    return [
        QualCitation(
            claim=claim,
            ticker=ticker,
            accession=str(h.payload.get("accession", "")),
            section=str(h.payload.get("section", "")),
            chunk_id=str(h.payload.get("chunk_id", h.chunk_id)),
            quote=str(h.payload.get("text", ""))[:240],
        )
        for h in hits
    ]


def analyze_axis(
    task: AxisTask,
    *,
    sonnet_client: ChatClient,
    pool: ConnectionPool,
    qdrant: QdrantClient,
    embed_query: Callable[[str], tuple[list[float], dict[str, float]]],
    branch_timeout_s: float = DEFAULT_BRANCH_TIMEOUT_S,
    prompt_version: str = "analyst-v1",
    llm_cache: Cache | None = None,
    trace_ctx: TraceCtx | None = None,
) -> AxisResult:
    # Node contract (Phase 6d):
    # - One Sonnet call. Code drives retrieval; the model just summarizes.
    # - In-node retry up to MAX_RETRIES on transient errors.
    # - Never raises; failures surface as grounding=insufficient.
    deadline = time.monotonic() + branch_timeout_s
    quant_evidence: dict[str, list[ResolvedFact]] = {}
    qual_evidence: dict[str, list[SearchHit]] = {}

    if task.mode == "structured":
        quant_evidence = _gather_structured_evidence(task, pool=pool)
    else:
        qual_evidence = _gather_semantic_evidence(task, qdrant=qdrant, embed_query=embed_query)

    if time.monotonic() >= deadline:
        return _insufficient(task, "branch_timeout", attempts=0)

    return _write_axis_result(
        task,
        quant_evidence=quant_evidence,
        qual_evidence=qual_evidence,
        sonnet_client=sonnet_client,
        deadline=deadline,
        prompt_version=prompt_version,
        llm_cache=llm_cache,
        trace_ctx=trace_ctx,
    )


def _write_axis_result(
    task: AxisTask,
    *,
    quant_evidence: dict[str, list[ResolvedFact]],
    qual_evidence: dict[str, list[SearchHit]],
    sonnet_client: ChatClient,
    deadline: float,
    prompt_version: str = "analyst-v1",
    llm_cache: Cache | None = None,
    trace_ctx: TraceCtx | None = None,
) -> AxisResult:
    # The write-and-cite phase: one Sonnet call over already-gathered evidence,
    # with retry, parse, and code-built citations. Shared by the single-shot
    # analyst (planner-gathered evidence) and the agentic analyst (legwork-gathered
    # evidence) so both cite only what was gathered and produce identical shapes.
    if task.mode == "structured":
        evidence_text = "".join(_format_quant_evidence(t, f) for t, f in quant_evidence.items())
    else:
        evidence_text = "".join(_format_qual_evidence(t, h) for t, h in qual_evidence.items())

    user = (
        f"axis: {task.axis}\n"
        f"mode: {task.mode}\n"
        f"tickers: {', '.join(task.tickers)}\n\n"
        "EVIDENCE\n--------\n"
        f"{evidence_text}\n"
        "Write the JSON object now."
    )

    last_err: Exception | None = None
    attempts = 0
    while attempts < MAX_RETRIES and time.monotonic() < deadline:
        attempts += 1
        try:
            resp = chat_maybe_cached(
                sonnet_client,
                llm_cache,
                prompt_version=prompt_version,
                system=[
                    {
                        "type": "text",
                        "text": _ANALYST_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=2000,
            )
            if trace_ctx is not None:
                trace_ctx.event(
                    "llm:analyst",
                    **llm_trace_fields(sonnet_client.model, resp),
                    input_shape={"axis": task.axis, "attempt": attempts},
                )
            raw = _extract_text(resp)
            parsed = _parse_analyst_output(raw)
            return _to_result(task, parsed, quant_evidence, qual_evidence, attempts)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5 * (2 ** (attempts - 1)))
            continue

    reason = f"transient_failure: {type(last_err).__name__}" if last_err else "branch_timeout"
    return _insufficient(task, reason, attempts=attempts)


def _to_result(
    task: AxisTask,
    parsed: dict[str, Any],
    quant: dict[str, list[ResolvedFact]],
    qual: dict[str, list[SearchHit]],
    attempts: int,
) -> AxisResult:
    grounding: Grounding = parsed.get("grounding", "ok")
    if grounding not in ("ok", "weak", "insufficient"):
        grounding = "weak"

    raw_assessments = parsed.get("per_company") or {}
    per_company: dict[str, CompanyAxisFinding] = {}
    citations: list[Citation] = []
    for ticker in task.tickers:
        # Narrative is the model's; numbers (values/citations) are code-built from
        # the authoritative facts so the model never re-types figures (that echo
        # overflowed max_tokens) and code-built numbers cannot drift.
        assessment = str(raw_assessments.get(ticker, "")).strip()
        if ticker in quant:
            facts = quant[ticker]
            values = {f"{f.resolved_concept}_{f.period}": f"{f.value} {f.unit}" for f in facts}
            passages: list[str] = []
            cits = _citations_for_quant(ticker, facts, claim=parsed.get("comparison", ""))
        else:
            hits = qual.get(ticker, [])
            values = {}
            passages = [str(h.payload.get("text", ""))[:240] for h in hits]
            cits = _citations_for_qual(ticker, hits, claim=parsed.get("comparison", ""))
        per_company[ticker] = CompanyAxisFinding(
            ticker=ticker,
            assessment=assessment,
            values=values,
            passages=passages,
            citations=cits,
        )
        citations.extend(cits)

    return AxisResult(
        axis=task.axis,
        mode=task.mode,
        per_company=per_company,
        comparison=str(parsed.get("comparison", "")),
        citations=citations,
        grounding=grounding,
        attempts=attempts,
        error_kind="none"
        if grounding == "ok"
        else ("terminal" if grounding == "insufficient" else "none"),
        error_reason=parsed.get("reason_if_not_ok") or None,
    )


def _insufficient(task: AxisTask, reason: str, *, attempts: int) -> AxisResult:
    return AxisResult(
        axis=task.axis,
        mode=task.mode,
        per_company={t: CompanyAxisFinding(ticker=t) for t in task.tickers},
        comparison="",
        citations=[],
        grounding="insufficient",
        attempts=attempts,
        error_kind="terminal",
        error_reason=reason,
    )


# ---- Phase 13c: tiered agentic analyst (flaggable, off by default) ----

DEFAULT_LEGWORK_TURNS = 5
DEFAULT_LEGWORK_WALL_CLOCK_S = 45.0

_LEGWORK_SYSTEM_TEXT = (
    "You are a Quorum research assistant. Gather the evidence an analyst needs to "
    "compare the given companies on one axis: pull the relevant XBRL facts with "
    "get_financial_concept and the relevant filing passages with search_filings "
    "(use get_filing_section to read more of a section when a snippet is not "
    "enough). Cover every ticker. Be efficient. When you have enough, stop with a "
    "one-line summary of what you gathered - do NOT write the analysis yourself."
)


class _LegworkDispatch:
    # Executes the legwork agent's tool calls and accumulates the typed evidence
    # (facts + hits) so the downstream write phase cites ONLY what was gathered.
    def __init__(
        self,
        *,
        pool: ConnectionPool,
        qdrant: QdrantClient,
        embed_query: Callable[[str], tuple[list[float], dict[str, float]]],
        trace_ctx: TraceCtx | None = None,
    ) -> None:
        self.pool = pool
        self.qdrant = qdrant
        self.embed_query = embed_query
        self.trace_ctx = trace_ctx
        self.facts: dict[str, list[ResolvedFact]] = {}
        self.hits: dict[str, list[SearchHit]] = {}

    def call(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        start = time.monotonic()
        ok, content = self._dispatch(name, args)
        if self.trace_ctx is not None:
            self.trace_ctx.event(
                f"tool:{name}",
                duration_ms=int((time.monotonic() - start) * 1000),
                error_kind="none" if ok else "transient",
                error_reason=None if ok else content[:240],
            )
        return ok, content

    def _dispatch(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        try:
            if name == "get_financial_concept":
                ticker = str(args["ticker"])
                facts = get_financial_concept(
                    self.pool, ticker=ticker, key=str(args["key"]), periods=args.get("periods")
                )
                self.facts.setdefault(ticker, []).extend(facts)
                return True, json.dumps(
                    [
                        {
                            "value": f.value,
                            "unit": f.unit,
                            "period": f.period,
                            "accession": f.accession,
                            "resolved_concept": f.resolved_concept,
                        }
                        for f in facts
                    ]
                )
            if name == "search_filings":
                dense, sparse = self.embed_query(str(args["query"]))
                hits = hybrid_search(
                    self.qdrant,
                    dense_vec=dense,
                    sparse_weights=sparse,
                    tickers=args.get("tickers"),
                    sections=args.get("sections"),
                    top_k=int(args.get("top_k", 5)),
                )
                for h in hits:
                    t = str(h.payload.get("ticker", ""))
                    if t:
                        self.hits.setdefault(t, []).append(h)
                return True, json.dumps(
                    [
                        {
                            "chunk_id": h.chunk_id,
                            "ticker": h.payload.get("ticker"),
                            "section": h.payload.get("section"),
                            "score": h.score,
                            "text_excerpt": str(h.payload.get("text", ""))[:400],
                        }
                        for h in hits
                    ]
                )
            if name == "get_filing_section":
                section = get_filing_section(
                    self.qdrant,
                    ticker=str(args["ticker"]),
                    accession=str(args["accession"]),
                    section=str(args["section"]),
                )
                return True, json.dumps({"text": section.text[:6000]})
            return False, f"unknown tool: {name}"
        except Exception as e:  # noqa: BLE001
            return False, f"tool_error: {type(e).__name__}: {e}"

    def evidence(
        self, task: AxisTask
    ) -> tuple[dict[str, list[ResolvedFact]], dict[str, list[SearchHit]]]:
        # Shape the accumulated evidence like the single-shot gather: keyed by the
        # task's tickers, deduped, quant trimmed to recent fiscal years.
        if task.mode == "structured":
            quant: dict[str, list[ResolvedFact]] = {}
            for t in task.tickers:
                seen: set[tuple[str, str, str]] = set()
                uniq: list[ResolvedFact] = []
                for f in self.facts.get(t, []):
                    key = (f.resolved_concept, f.period, f.unit)
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(f)
                quant[t] = _trim_to_recent_fiscal_years(uniq)
            return quant, {}
        qual: dict[str, list[SearchHit]] = {}
        for t in task.tickers:
            seen_ids: set[str] = set()
            uniq_hits: list[SearchHit] = []
            for h in self.hits.get(t, []):
                cid = str(h.payload.get("chunk_id", h.chunk_id))
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                uniq_hits.append(h)
            qual[t] = uniq_hits[:8]
        return {}, qual


def _has_any_evidence(
    quant: dict[str, list[ResolvedFact]], qual: dict[str, list[SearchHit]]
) -> bool:
    return any(quant.values()) or any(qual.values())


def analyze_axis_agentic(
    task: AxisTask,
    *,
    legwork_client: ChatClient,
    sonnet_client: ChatClient,
    pool: ConnectionPool,
    qdrant: QdrantClient,
    embed_query: Callable[[str], tuple[list[float], dict[str, float]]],
    branch_timeout_s: float = DEFAULT_BRANCH_TIMEOUT_S,
    legwork_turns: int = DEFAULT_LEGWORK_TURNS,
    legwork_wall_clock_s: float = DEFAULT_LEGWORK_WALL_CLOCK_S,
    prompt_version: str = "analyst-v1",
    legwork_prompt_version: str = "legwork-v1",
    llm_cache: Cache | None = None,
    trace_ctx: TraceCtx | None = None,
) -> AxisResult:
    # Phase 13c, flaggable. A cheap legwork agent gathers evidence via tools, then
    # one Sonnet write-and-cite pass over only that evidence (parity with the
    # single-shot write by construction). Any failure or empty legwork falls back
    # to the single-shot analyst, which is also the A/B baseline. Never raises.
    def _fallback() -> AxisResult:
        return analyze_axis(
            task,
            sonnet_client=sonnet_client,
            pool=pool,
            qdrant=qdrant,
            embed_query=embed_query,
            branch_timeout_s=branch_timeout_s,
            prompt_version=prompt_version,
            llm_cache=llm_cache,
            trace_ctx=trace_ctx,
        )

    try:
        dispatch = _LegworkDispatch(
            pool=pool, qdrant=qdrant, embed_query=embed_query, trace_ctx=trace_ctx
        )
        initial_user = (
            f"axis: {task.axis}\n"
            f"mode: {task.mode}\n"
            f"tickers: {', '.join(task.tickers)}\n"
            f"focus: {task.query_or_concept}\n\n"
            "Gather evidence now."
        )
        run_agent_loop(
            client=legwork_client,
            system=[
                {
                    "type": "text",
                    "text": _LEGWORK_SYSTEM_TEXT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=_LEGWORK_TOOL_DEFS,
            dispatch=dispatch.call,
            initial_user=initial_user,
            max_turns=legwork_turns,
            wall_clock_s=legwork_wall_clock_s,
            label="legwork",
            llm_cache=llm_cache,
            prompt_version=legwork_prompt_version,
            trace_ctx=trace_ctx,
        )
        quant_evidence, qual_evidence = dispatch.evidence(task)
        if not _has_any_evidence(quant_evidence, qual_evidence):
            return _fallback()
        return _write_axis_result(
            task,
            quant_evidence=quant_evidence,
            qual_evidence=qual_evidence,
            sonnet_client=sonnet_client,
            deadline=time.monotonic() + branch_timeout_s,
            prompt_version=prompt_version,
            llm_cache=llm_cache,
            trace_ctx=trace_ctx,
        )
    except Exception:  # noqa: BLE001
        return _fallback()
