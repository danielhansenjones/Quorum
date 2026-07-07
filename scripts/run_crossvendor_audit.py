from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path

from diskcache import Cache
from qdrant_client import QdrantClient

from quorum.config.settings import get_settings
from quorum.eval.judge_correlation import correlate
from quorum.eval.judges import (
    reconstruct_citation,
    score_report_quality,
    verify_qual_citation,
)
from quorum.models.router import get_client
from quorum.state.citation import QualCitation

# Cross-vendor self-preference audit. Sonnet writes the reports and Sonnet judged
# them, so the canonical scores could be inflated by same-family preference.
# Re-judge the same reports and qual claims with a pinned GPT snapshot (reasoning
# off, so it is a plain non-reasoning judge comparable to the non-reasoning Sonnet
# judge) and compare. Sonnet's side is read from the committed artifacts - no
# Anthropic re-spend. The audit judge comes from settings (AUDIT_JUDGE_MODEL, a
# pinned dated snapshot) via the gated judge_audit role. One pass, cached,
# cost-capped; dry-run cost estimate unless --yes.
#
# campaign-critic is the default-config judged run (critic on, both arms off) and
# the only committed set that carries Sonnet-judged qual claims; judged-full-v1-
# final stores faithfulness as quant-only, so the faithfulness half cannot run
# there.

DEFAULT_RUN_DIR = Path("eval/results/campaign-critic")
DEFAULT_OUT = Path("eval/results/crossvendor/audit.json")
DEFAULT_CACHE = Path("data/cache/crossvendor")  # under the gitignored data/cache tree
DEFAULT_MAX_USD = 5.0

# gpt-5.1 Chat Completions pricing, USD/token (verified 2026-07; update if it moves).
PRICE_IN = 1.25 / 1_000_000
PRICE_OUT = 10.0 / 1_000_000
CHARS_PER_TOK = 4  # English-prose approximation for the pre-run estimate only.


def _quality_mean(quality: dict | None) -> float | None:
    if not quality:
        return None
    ints = [v for v in quality.values() if isinstance(v, int)]
    return sum(ints) / len(ints) if ints else None


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _sonnet_qual_faith(scores: dict | None) -> float | None:
    claims = ((scores or {}).get("faithfulness") or {}).get("claims") or []
    ss = [
        c["score"] for c in claims if c.get("judge") == "sonnet" and isinstance(c.get("score"), int)
    ]
    return _mean(ss)


def _eligible(d: dict) -> bool:
    return d.get("final_status") in ("ok", "partial") and bool((d.get("report") or "").strip())


def _collect(run_dir: Path, limit: int | None) -> list[tuple[str, Path, dict]]:
    files = [
        p
        for p in sorted(run_dir.glob("*.json"))
        if p.name not in ("summary.json", "cost_report.json")
    ]
    out: list[tuple[str, Path, dict]] = []
    for p in files:
        d = json.loads(p.read_text())
        if not _eligible(d):
            continue
        out.append((d.get("case_id") or p.stem, p, d))
        if limit is not None and len(out) >= limit:
            break
    return out


def _estimate(cases: list[tuple[str, Path, dict]], scope: str) -> dict:
    n_quality = 0
    n_faith = 0
    in_chars = 0
    for _, _, d in cases:
        report = d.get("report") or ""
        if scope in ("quality", "both"):
            n_quality += 1
            in_chars += 260 + min(len(report), 6000)
        if scope in ("faithfulness", "both"):
            for c in d.get("citations") or []:
                if c.get("kind") == "qual":
                    n_faith += 1
                    in_chars += 640 + len(c.get("claim") or "") + 4000
    in_tok = in_chars / CHARS_PER_TOK
    calls = n_quality + n_faith
    # reasoning_effort=none: visible JSON verdict dominates, reasoning is minimal.
    # Bracket low (near-zero reasoning) to a padded ceiling.
    lo_out = calls * (150 + 50)
    hi_out = calls * (150 + 700)
    return {
        "quality_calls": n_quality,
        "faithfulness_calls": n_faith,
        "total_calls": calls,
        "input_tokens_est": int(in_tok),
        "cost_low": round(in_tok * PRICE_IN + lo_out * PRICE_OUT, 2),
        "cost_high": round(in_tok * PRICE_IN + hi_out * PRICE_OUT, 2),
    }


def _signed_delta(pairs: list[tuple[float, float]]) -> dict:
    # Positive mean delta = Sonnet scores its own reports higher than the GPT
    # referee does = self-preference; a ci95 that straddles 0 means no detectable
    # preference. Normal-approx is valid because the audit is single-arm: one
    # report per question, so the per-case deltas are independent (small faith n
    # keeps its interval wide - read it as directional).
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    deltas = [s - g for s, g in pairs]
    mean = sum(deltas) / n
    out = {
        "n": n,
        "sonnet_mean": round(sum(s for s, _ in pairs) / n, 4),
        "gpt_mean": round(sum(g for _, g in pairs) / n, 4),
        "mean_delta_sonnet_minus_gpt": round(mean, 4),
        "ci95": None,
    }
    if n > 1:
        sem = sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1)) / sqrt(n)
        out["ci95"] = [round(mean - 1.96 * sem, 4), round(mean + 1.96 * sem, 4)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cross-vendor self-preference audit: re-judge the Sonnet-written gold set with "
        "a pinned GPT snapshot (reasoning off) and compare to Sonnet's committed scores. "
        "Dry-run cost estimate unless --yes."
    )
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--scope", choices=("quality", "faithfulness", "both"), default="both")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument(
        "--max-usd",
        type=float,
        default=DEFAULT_MAX_USD,
        help="Hard cap. If the one-pass estimate exceeds it, stop before spending.",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="First N eligible cases (cheap probe)."
    )
    ap.add_argument("--yes", action="store_true", help="Actually spend. Without it, estimate only.")
    args = ap.parse_args()

    cases = _collect(args.run_dir, args.limit)
    if not cases:
        print(f"no eligible cases under {args.run_dir}", file=sys.stderr)
        return 1
    est = _estimate(cases, args.scope)
    print(json.dumps({"estimate": est, "scope": args.scope, "n_cases": len(cases)}, indent=2))

    if not args.yes:
        print(">>> dry run. Re-run with --yes to spend.", file=sys.stderr)
        return 0

    if est["cost_high"] > args.max_usd:
        print(
            f">>> estimated ${est['cost_high']} exceeds cap ${args.max_usd}; "
            "raise --max-usd or narrow with --limit / --scope. Not spending.",
            file=sys.stderr,
        )
        return 2

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; run via ./secret-run.", file=sys.stderr)
        return 2
    if not get_settings().audit_judge_model:
        print(
            "AUDIT_JUDGE_MODEL not set; export a pinned GPT-5.1 snapshot (e.g. gpt-5.1-2025-11-13).",
            file=sys.stderr,
        )
        return 2

    audit = get_client("judge_audit")
    cache = Cache(str(args.cache))
    qdrant = QdrantClient(url=get_settings().qdrant_url) if args.scope != "quality" else None

    quality_pairs: list[tuple[float, float]] = []
    quality_labels: list[str] = []
    faith_pairs: list[tuple[float, float]] = []
    faith_labels: list[str] = []
    per_case: list[dict] = []
    quality_failures = 0
    faith_failures = 0

    for i, (cid, _path, d) in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {cid}", file=sys.stderr)
        scores = d.get("scores") or {}
        report = d.get("report") or ""
        row: dict = {"case": cid}

        if args.scope in ("quality", "both"):
            son_q = _quality_mean(scores.get("quality"))
            gpt_quality = score_report_quality(report, judge_client=audit, llm_cache=cache)
            gpt_q = _quality_mean(gpt_quality)
            if gpt_quality and gpt_q is None:
                quality_failures += 1
            row |= {"sonnet_quality": son_q, "gpt_quality": gpt_q}
            if son_q is not None and gpt_q is not None:
                quality_pairs.append((son_q, gpt_q))
                quality_labels.append(cid)

        if args.scope in ("faithfulness", "both"):
            assert qdrant is not None
            quals = [
                c
                for c in (reconstruct_citation(x) for x in (d.get("citations") or []))
                if isinstance(c, QualCitation)
            ]
            gpt_v = [
                verify_qual_citation(qdrant, c, judge_client=audit, llm_cache=cache) for c in quals
            ]
            # Exclude unparseable verdicts rather than folding them in as score=1,
            # which would bias the delta toward false self-preference.
            gpt_scored = [v for v in gpt_v if v.judge != "error"]
            faith_failures += len(gpt_v) - len(gpt_scored)
            son_f = _sonnet_qual_faith(scores)
            gpt_f = _mean([v.score for v in gpt_scored])
            row |= {"sonnet_faith_qual_only": son_f, "gpt_faith_qual_only": gpt_f}
            if son_f is not None and gpt_f is not None:
                faith_pairs.append((son_f, gpt_f))
                faith_labels.append(cid)

        per_case.append(row)

    quality_corr = correlate(
        [s for s, _ in quality_pairs], [g for _, g in quality_pairs], labels=quality_labels
    )
    faith_corr = correlate(
        [s for s, _ in faith_pairs], [g for _, g in faith_pairs], labels=faith_labels
    )
    spent = audit.usage["prompt"] * PRICE_IN + audit.usage["completion"] * PRICE_OUT

    summary = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "audit_judge": audit.model,
        "reasoning_effort": audit.reasoning_effort,
        "baseline": "claude-sonnet-4-6 (read from committed gold-set artifacts)",
        "run_dir": str(args.run_dir),
        "n_cases": len(cases),
        "quality_parse_failures": quality_failures,
        "faithfulness_parse_failures": faith_failures,
        "quality": {"correlation": quality_corr, "self_preference": _signed_delta(quality_pairs)},
        "faithfulness_qual_only": {
            "correlation": faith_corr,
            "self_preference": _signed_delta(faith_pairs),
        },
        "usage": audit.usage,
        "actual_cost_usd": round(spent, 4),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}  (actual spend ${spent:.4f})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
