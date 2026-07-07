from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from qdrant_client import QdrantClient

from quorum.config.settings import get_settings
from quorum.eval.judge_correlation import correlate, judge_decision
from quorum.eval.judges import (
    reconstruct_citation,
    score_report_quality,
    verify_qual_citation,
    verify_quant_citation,
)
from quorum.models.router import DEFAULT_VLLM_MODEL, get_client
from quorum.state.citation import QualCitation, QuantCitation
from quorum.trace.writer import open_pool

# All four campaign arms of the held-out questions. Same base case_ids, but each
# arm is a different report, so quality n grows without touching the leakage
# split (the split is by base case_id, so every arm of a val question is held
# out). Restrict to the val ids with --only-cases for the honest gate.
DEFAULT_RUN_DIRS = [
    Path("eval/results/campaign-baseline"),
    Path("eval/results/campaign-agentic"),
    Path("eval/results/campaign-critic"),
    Path("eval/results/campaign-rebuttal"),
]
DEFAULT_OUT = Path("eval/judge_config.yaml")
DEFAULT_PAIRS = Path("eval/results/judge_correlation/study.json")
DEFAULT_VLLM = "http://localhost:8001/v1"


def _quality_mean(quality: dict | None) -> float | None:
    if not quality:
        return None
    ints = [v for v in quality.values() if isinstance(v, int)]
    return sum(ints) / len(ints) if ints else None


def _mean_score(verdicts: list) -> float | None:
    return sum(v.score for v in verdicts) / len(verdicts) if verdicts else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 10c: local 7B vs canonical Sonnet judge correlation"
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        default=DEFAULT_RUN_DIRS,
        help="Run dirs to pull reports from. Default is all four held-out arms so quality n "
        "counts every distinct report, not one per question.",
    )
    parser.add_argument("--vllm-url", type=str, default=DEFAULT_VLLM)
    parser.add_argument(
        "--vllm-model",
        type=str,
        default=DEFAULT_VLLM_MODEL,
        help="Served model for the local judge. Set to the LoRA adapter (e.g. judge-qlora) to gate the fine-tune.",
    )
    parser.add_argument(
        "--only-cases",
        type=Path,
        default=None,
        help="File of case_ids (one per line) to restrict scoring to; point at the SFT val split for a held-out gate.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--pairs-out",
        type=Path,
        default=DEFAULT_PAIRS,
        help="Tracked path for the raw per-case score pairs behind the correlations.",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Needs Sonnet for the canonical qual re-judge; run via ./secret-run.")
        return 2

    settings = get_settings()
    pool = open_pool(conninfo=settings.postgres_url, min_size=1, max_size=settings.pg_pool_max)
    qdrant = QdrantClient(url=settings.qdrant_url)
    local_judge = get_client("judge_dev", vllm_url=args.vllm_url, vllm_model=args.vllm_model)
    if local_judge.backend != "vllm":
        print(f"local judge resolved to {local_judge.backend}, not vllm; pass --vllm-url")
        return 2
    sonnet_judge = get_client("judge_canonical")

    # Faithfulness is correlated three ways. Blended = mean over all citations,
    # but quant citations are scored by identical deterministic code on both
    # judges, so that shared floor inflates the rank correlation toward 1.
    # Qual-only isolates the part the LLM judge actually decides. Within qual-
    # only, the per-case mean collapses every question to a single point (n=3);
    # the per-claim view keeps each claim the judge scored (n in the hundreds).
    # Each pair carries its base case_id so the CI can resample by question and
    # not overstate independence. The claim-level lower bound is the gate.
    blended_pairs: list[tuple[float, float]] = []
    blended_labels: list[str] = []
    qual_case_pairs: list[tuple[float, float]] = []
    qual_case_labels: list[str] = []
    qual_claim_pairs: list[tuple[float, float]] = []
    qual_claim_labels: list[str] = []
    quality_pairs: list[tuple[float, float]] = []
    quality_labels: list[str] = []
    per_case: list[dict] = []
    local_quality_failures = 0

    case_files = [
        p
        for rd in args.run_dirs
        for p in sorted(rd.glob("*.json"))
        if p.name not in ("summary.json", "cost_report.json")
    ]
    allowed = None
    if args.only_cases:
        allowed = {ln.strip() for ln in args.only_cases.read_text().splitlines() if ln.strip()}
    for i, p in enumerate(case_files, start=1):
        d = json.loads(p.read_text())
        cid = d.get("case_id") or p.stem
        arm = p.parent.name
        if allowed is not None and cid not in allowed:
            continue
        if d.get("final_status") not in ("ok", "partial"):
            continue
        report = d.get("report") or ""
        if not report.strip():
            continue
        # Progress to stderr; stdout stays parseable JSON.
        print(f"[{i}/{len(case_files)}] {arm}/{cid}", file=sys.stderr)

        recon = [reconstruct_citation(x) for x in (d.get("citations") or [])]
        quant = [verify_quant_citation(pool, c) for c in recon if isinstance(c, QuantCitation)]
        quals = [c for c in recon if isinstance(c, QualCitation)]
        loc_q = [verify_qual_citation(qdrant, c, judge_client=local_judge) for c in quals]
        son_q = [verify_qual_citation(qdrant, c, judge_client=sonnet_judge) for c in quals]

        loc_qual_faith = _mean_score(loc_q)
        son_qual_faith = _mean_score(son_q)
        loc_blended = _mean_score(quant + loc_q)
        # Sonnet blended is read from the canonical run (matches the published
        # 4.51); quant verdicts are deterministic, so only qual is re-judged.
        son_blended = ((d.get("scores") or {}).get("faithfulness") or {}).get("mean_score")

        local_quality = score_report_quality(report, judge_client=local_judge)
        loc_qual = _quality_mean(local_quality)
        if local_quality and loc_qual is None:
            local_quality_failures += 1
        son_qual = _quality_mean((d.get("scores") or {}).get("quality"))

        if son_blended is not None and loc_blended is not None:
            blended_pairs.append((son_blended, loc_blended))
            blended_labels.append(cid)
        if son_qual_faith is not None and loc_qual_faith is not None:
            qual_case_pairs.append((son_qual_faith, loc_qual_faith))
            qual_case_labels.append(cid)
        for sv, lv in zip(son_q, loc_q, strict=True):
            qual_claim_pairs.append((sv.score, lv.score))
            qual_claim_labels.append(cid)
        if son_qual is not None and loc_qual is not None:
            quality_pairs.append((son_qual, loc_qual))
            quality_labels.append(cid)

        per_case.append(
            {
                "case": cid,
                "arm": arm,
                "n_qual_citations": len(quals),
                "sonnet_faith_blended": son_blended,
                "local_faith_blended": loc_blended,
                "sonnet_faith_qual_only": son_qual_faith,
                "local_faith_qual_only": loc_qual_faith,
                "sonnet_quality": son_qual,
                "local_quality": loc_qual,
            }
        )

    def _split(pairs: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
        return [a for a, _ in pairs], [b for _, b in pairs]

    blended_corr = correlate(*_split(blended_pairs), labels=blended_labels)
    qual_case_corr = correlate(*_split(qual_case_pairs), labels=qual_case_labels)
    qual_claim_corr = correlate(*_split(qual_claim_pairs), labels=qual_claim_labels)
    quality_corr = correlate(*_split(quality_pairs), labels=quality_labels)
    # Gate on the honest signal: qual-only faithfulness at the claim level, with a
    # CI clustered by question, not the inflated blend or the collapsed per-case mean.
    decision = judge_decision(qual_claim_corr, quality_corr)

    config = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dirs": [str(rd) for rd in args.run_dirs],
        "local_judge": local_judge.model,
        "canonical_judge": sonnet_judge.model,
        "local_quality_parse_failures": local_quality_failures,
        "faithfulness_qual_only_per_claim": qual_claim_corr,
        "faithfulness_qual_only_per_case": qual_case_corr,
        "faithfulness_blended": blended_corr,
        "quality": quality_corr,
        "decision": decision,
    }
    args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    study = {"config": config, "per_case": per_case}
    args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_out.write_text(json.dumps(study, indent=2))
    print(json.dumps(study, indent=2))
    print(f"wrote {args.pairs_out}")
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
