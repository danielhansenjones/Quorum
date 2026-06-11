from __future__ import annotations

import argparse
import json
import os
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
from quorum.models.router import get_client
from quorum.state.citation import QualCitation, QuantCitation
from quorum.trace.writer import open_pool

DEFAULT_RUN = Path("eval/runs/judged-full-v1-final")
DEFAULT_OUT = Path("eval/judge_config.yaml")
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
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--vllm-url", type=str, default=DEFAULT_VLLM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Needs Sonnet for the canonical qual re-judge; run via ./secret-run.")
        return 2

    settings = get_settings()
    pool = open_pool(conninfo=settings.postgres_url, min_size=1, max_size=settings.pg_pool_max)
    qdrant = QdrantClient(url=settings.qdrant_url)
    local_judge = get_client("judge_dev", vllm_url=args.vllm_url)
    if local_judge.backend != "vllm":
        print(f"local judge resolved to {local_judge.backend}, not vllm; pass --vllm-url")
        return 2
    sonnet_judge = get_client("judge_canonical")

    # Faithfulness is correlated two ways. Blended = mean over all citations, but
    # quant citations are scored by identical deterministic code on both judges,
    # so that shared floor inflates the rank correlation toward 1. Qual-only
    # isolates the part the LLM judge actually decides, and is the honest gate.
    blended_pairs: list[tuple[float, float]] = []
    qual_pairs: list[tuple[float, float]] = []
    quality_pairs: list[tuple[float, float]] = []
    per_case: list[dict] = []
    local_quality_failures = 0

    for p in sorted(args.run_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text())
        if d.get("final_status") not in ("ok", "partial"):
            continue
        report = d.get("report") or ""
        if not report.strip():
            continue

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
        if son_qual_faith is not None and loc_qual_faith is not None:
            qual_pairs.append((son_qual_faith, loc_qual_faith))
        if son_qual is not None and loc_qual is not None:
            quality_pairs.append((son_qual, loc_qual))

        per_case.append(
            {
                "case": d["case_id"],
                "n_qual_citations": len(quals),
                "sonnet_faith_blended": son_blended,
                "local_faith_blended": loc_blended,
                "sonnet_faith_qual_only": son_qual_faith,
                "local_faith_qual_only": loc_qual_faith,
                "sonnet_quality": son_qual,
                "local_quality": loc_qual,
            }
        )

    blended_corr = correlate([a for a, _ in blended_pairs], [b for _, b in blended_pairs])
    qual_corr = correlate([a for a, _ in qual_pairs], [b for _, b in qual_pairs])
    quality_corr = correlate([a for a, _ in quality_pairs], [b for _, b in quality_pairs])
    # Gate on the honest signal: qual-only faithfulness, not the inflated blend.
    decision = judge_decision(qual_corr, quality_corr)

    config = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subset": str(args.run_dir),
        "local_judge": local_judge.model,
        "canonical_judge": sonnet_judge.model,
        "local_quality_parse_failures": local_quality_failures,
        "faithfulness_qual_only": qual_corr,
        "faithfulness_blended": blended_corr,
        "quality": quality_corr,
        "decision": decision,
    }
    args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"config": config, "per_case": per_case}, indent=2))
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
