from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_provenance() -> dict[str, Any]:
    # Ties a run dir to the code that produced it. A dirty tree is recorded, not
    # rejected: subset and debug runs are legitimate, but a campaign artifact
    # with git_dirty=true is visibly not canonical. None outside a git checkout.
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"git_sha": sha, "git_dirty": bool(status) if status is not None else None}


@dataclass(frozen=True, slots=True)
class GoldCase:
    id: str
    question: str
    expected_status: str
    expected_axes: list[str]
    expected_tickers: list[str]
    notes: str = ""


@dataclass
class CaseResult:
    case_id: str
    request_id: str
    final_status: str
    report: str
    citations: list[dict[str, Any]]
    elapsed_s: float
    trajectory: list[str] = field(default_factory=list)
    critique: dict[str, Any] | None = None
    rebuttals: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_gold(path: Path) -> list[GoldCase]:
    raw = yaml.safe_load(path.read_text()) or {}
    return [
        GoldCase(
            id=c["id"],
            question=c["question"],
            expected_status=c.get("expected_status", "ok"),
            expected_axes=list(c.get("expected_axes") or []),
            expected_tickers=list(c.get("expected_tickers") or []),
            notes=c.get("notes", ""),
        )
        for c in raw.get("cases", [])
    ]


def _get(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _critique_to_dict(c: Any) -> dict[str, Any] | None:
    if c is None:
        return None
    if isinstance(c, dict):
        return c
    dump = getattr(c, "model_dump", None)
    return dict(dump(mode="json")) if callable(dump) else None


def _rebuttals_to_list(rebuttals: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rb in rebuttals or []:
        if isinstance(rb, dict):
            out.append(rb)
            continue
        dump = getattr(rb, "model_dump", None)
        if callable(dump):
            out.append(dict(dump(mode="json")))
    return out


def _flagged_claims_from_critique(critique: dict[str, Any] | None) -> list[Any] | None:
    # None when no critique was captured (critic off/failed); a (possibly empty)
    # list otherwise, so the incorporation scorer can distinguish "no flags" from
    # "no critic". Malformed entries are skipped, not raised.
    if critique is None:
        return None
    from quorum.state.critique import FlaggedClaim

    out: list[Any] = []
    for fc in critique.get("flagged_claims", []) or []:
        try:
            out.append(FlaggedClaim(**fc))
        except Exception:  # noqa: BLE001
            continue
    return out


def run_case(case: GoldCase, *, compiled_graph: Any) -> CaseResult:
    from quorum.graph.build import initial_state

    start = time.monotonic()
    state = initial_state(case.question)
    try:
        # Single pass: "updates" gives the node-traversal trajectory, "values"
        # gives the accumulated state (the last chunk is final). Two passes would
        # double the cost - the same bug the API already fixed by reading state once.
        trajectory: list[str] = []
        final: Any = None
        for mode, chunk in compiled_graph.stream(state, stream_mode=["updates", "values"]):
            if mode == "updates":
                trajectory.extend(chunk.keys())
            else:
                final = chunk
        citations = _get(final, "report_citations", []) or []
        return CaseResult(
            case_id=case.id,
            request_id=state.request_id,
            final_status=str(_get(final, "status", "ok")),
            report=str(_get(final, "report", None) or ""),
            citations=[c if isinstance(c, dict) else c.model_dump() for c in citations],
            trajectory=trajectory,
            critique=_critique_to_dict(_get(final, "critique", None)),
            rebuttals=_rebuttals_to_list(_get(final, "rebuttals", [])),
            elapsed_s=time.monotonic() - start,
        )
    except Exception as e:  # noqa: BLE001
        return CaseResult(
            case_id=case.id,
            request_id=state.request_id,
            final_status="error",
            report="",
            citations=[],
            elapsed_s=time.monotonic() - start,
            error=f"{type(e).__name__}: {e}",
        )


def run_all(
    gold_path: Path,
    *,
    compiled_graph: Any,
    out_dir: Path,
    pool: Any | None = None,
    qdrant: Any | None = None,
    judge_client: Any | None = None,
    llm_cache: Any | None = None,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Judging is opt-in: pass pool + qdrant + judge_client to score each case's
    # faithfulness (deterministic quant + LLM qual) and report quality.
    judge = pool is not None and qdrant is not None and judge_client is not None
    started_at = datetime.now(UTC)
    cases = load_gold(gold_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []
    for c in cases:
        r = run_case(c, compiled_graph=compiled_graph)
        if judge:
            from quorum.eval.judges import score_case

            assert pool is not None and qdrant is not None and judge_client is not None
            r.extra["scores"] = score_case(
                report=r.report,
                citations=r.citations,
                pool=pool,
                qdrant=qdrant,
                judge_client=judge_client,
                flagged_claims=_flagged_claims_from_critique(r.critique),
                llm_cache=llm_cache,
            )
        results.append(r)
        # Per-case JSON file; the comparison-stable form has timestamps stripped.
        (out_dir / f"{c.id}.json").write_text(
            json.dumps(
                {
                    "case_id": r.case_id,
                    "request_id": r.request_id,
                    "final_status": r.final_status,
                    "elapsed_s": r.elapsed_s,
                    "trajectory": r.trajectory,
                    "report": r.report,
                    "citations": r.citations,
                    "critique": r.critique,
                    "rebuttals": r.rebuttals,
                    "scores": r.extra.get("scores"),
                    "error": r.error,
                },
                indent=2,
            )
        )
    summary: dict[str, Any] = {
        "n_cases": len(cases),
        "ok": sum(1 for r in results if r.final_status == "ok"),
        "refused": sum(1 for r in results if r.final_status == "refused"),
        "partial": sum(1 for r in results if r.final_status == "partial"),
        "errors": sum(1 for r in results if r.final_status == "error"),
        "total_elapsed_s": sum(r.elapsed_s for r in results),
        "status_match": sum(
            1 for c, r in zip(cases, results, strict=True) if c.expected_status == r.final_status
        ),
    }
    if run_config is not None:
        # Records the toggle combination this arm ran under so the A/B campaign
        # can identify an arm from its run dir, not just its name (Phase 12e/13b).
        summary["run_config"] = run_config
    summary["provenance"] = {
        **_git_provenance(),
        "started_at": started_at.isoformat(),
        "judge_model": judge_client.model if judge and judge_client is not None else None,
    }
    if judge:
        summary["judging"] = _aggregate_judging(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _aggregate_judging(results: list[CaseResult]) -> dict[str, Any]:
    faith_means = [
        s["faithfulness"]["mean_score"]
        for r in results
        if (s := r.extra.get("scores")) and s["faithfulness"]["mean_score"] is not None
    ]
    quality_means: list[float] = []
    quality_judge_failures = 0
    for r in results:
        # A refusal has no report; scoring its message for comparison_quality /
        # evidence_coverage measures nothing and drags the mean down. Quality is
        # over produced reports only - mirrors faithfulness, which already skips
        # refusals (zero citations -> mean_score None).
        if r.final_status == "refused":
            continue
        s = r.extra.get("scores")
        if not s or not s.get("quality"):
            continue
        ints = [v for v in s["quality"].values() if isinstance(v, int)]
        if not ints:
            # quality dict carries only a judge_error marker / notes: the rubric
            # was unscorable. Count it; don't fold a 0 into the mean.
            quality_judge_failures += 1
            continue
        quality_means.append(sum(ints) / len(ints))
    return {
        "cases_scored": sum(1 for r in results if r.extra.get("scores")),
        "faithfulness_mean": (sum(faith_means) / len(faith_means)) if faith_means else None,
        "quality_mean": (sum(quality_means) / len(quality_means)) if quality_means else None,
        "quality_judge_failures": quality_judge_failures,
        "critic": _aggregate_critic(results),
    }


def _aggregate_critic(results: list[CaseResult]) -> dict[str, Any]:
    # Phase 12e. Critic headline metrics over the cases that produced a critique:
    # incorporation rate (judge-time), turns_used distribution, timeout rate, and
    # tool-use validity over the recorded critic tool calls.
    from quorum.eval.ab_compare import summarize_rebuttals
    from quorum.eval.tool_use import validate_tool_calls
    from quorum.state.critique import ToolCallRecord

    crit = [r for r in results if r.critique]
    n = len(crit)
    if n == 0:
        return {"cases_with_critique": 0}
    inc_total = inc_ok = timeouts = 0
    turns_counts: dict[int, int] = {}
    tool_records: list[ToolCallRecord] = []
    for r in crit:
        c = r.critique or {}
        scores = r.extra.get("scores") or {}
        if inc := scores.get("incorporation"):
            inc_total += int(inc.get("n") or 0)
            inc_ok += int(inc.get("incorporated") or 0)
        t = int(c.get("turns_used", 0))
        turns_counts[t] = turns_counts.get(t, 0) + 1
        if c.get("status") == "timeout":
            timeouts += 1
        for tc in c.get("tool_calls") or []:
            try:
                tool_records.append(ToolCallRecord(**tc))
            except Exception:  # noqa: BLE001
                continue
    tool_validity = validate_tool_calls(tool_records)
    rb_total = {"n": 0, "defended": 0, "retracted": 0, "revised": 0, "reflected": 0}
    cases_with_rebuttals = 0
    for r in results:
        if not r.rebuttals:
            continue
        cases_with_rebuttals += 1
        s = summarize_rebuttals(r.report, r.rebuttals)
        for k in rb_total:
            rb_total[k] += int(s[k])
    return {
        "cases_with_critique": n,
        "flagged_claims_total": inc_total,
        "incorporation_rate": (inc_ok / inc_total) if inc_total else None,
        "turns_used_counts": {str(k): turns_counts[k] for k in sorted(turns_counts)},
        "timeout_rate": timeouts / n,
        "tool_use": {
            "n": tool_validity["n"],
            "valid": tool_validity["valid"],
            "valid_fraction": tool_validity["valid_fraction"],
        },
        "rebuttal": {
            **rb_total,
            "cases": cases_with_rebuttals,
            "reflected_rate": (rb_total["reflected"] / rb_total["n"]) if rb_total["n"] else None,
        },
    }
