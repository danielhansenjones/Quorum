from __future__ import annotations

from typing import Any

# Phase 10c: does the local 7B judge agree with the canonical Sonnet judge well
# enough to use it for cheap iteration? Pure correlation + decision logic here;
# the driver (scripts/run_judge_correlation.py) does the IO and scoring.


def correlate(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) != len(y):
        raise ValueError("x and y must align 1:1")
    n = len(x)
    if n < 3:
        return {"n": n, "pearson": None, "spearman": None, "note": "n<3"}
    if len(set(x)) == 1 or len(set(y)) == 1:
        # A near-constant score vector (e.g. faithfulness pinned at 5.0 by
        # deterministic quant verdicts) makes correlation undefined, not 0.
        return {"n": n, "pearson": None, "spearman": None, "note": "constant input"}
    from scipy import stats  # eval extra; lazy so the base package needs no scipy

    return {
        "n": n,
        "pearson": float(stats.pearsonr(x, y).statistic),
        "spearman": float(stats.spearmanr(x, y).statistic),
    }


def judge_decision(
    faithfulness: dict[str, Any],
    quality: dict[str, Any],
    *,
    faith_threshold: float = 0.7,
    quality_threshold: float = 0.6,
) -> dict[str, Any]:
    sf = faithfulness.get("spearman")
    sq = quality.get("spearman")
    use_local = (
        sf is not None and sq is not None and sf > faith_threshold and sq > quality_threshold
    )
    return {
        "use_local_for_iteration": use_local,
        "rule": (
            f"local judge for iteration if spearman(faithfulness) > {faith_threshold} "
            f"AND spearman(quality) > {quality_threshold}; otherwise Sonnet for all judging"
        ),
        "faithfulness_spearman": sf,
        "quality_spearman": sq,
    }
