from __future__ import annotations

import math
from typing import Any

# Phase 10c: does the local 7B judge agree with the canonical Sonnet judge well
# enough to use it for cheap iteration? Pure correlation + decision logic here;
# the driver (scripts/run_judge_correlation.py) does the IO and scoring.
#
# The held-out set is small and clustered: a handful of base questions, each
# yielding many claim- and arm-level rows. A raw Spearman on those rows reads as
# more independent than it is, so the gate resamples whole clusters (base
# case_ids) to get a CI that reflects the true unit of independence, and decides
# on the lower bound rather than the point estimate.


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(set(x)) == 1 or len(set(y)) == 1:
        return None
    from scipy import stats

    s = float(stats.spearmanr(x, y).statistic)
    return None if math.isnan(s) else s


def _cluster_bootstrap_ci(
    x: list[float], y: list[float], labels: list[Any], *, n_boot: int, seed: int
) -> tuple[float | None, float | None, int]:
    import numpy as np

    groups: dict[Any, tuple[list[float], list[float]]] = {}
    for xi, yi, lab in zip(x, y, labels, strict=True):
        gx, gy = groups.setdefault(lab, ([], []))
        gx.append(xi)
        gy.append(yi)
    keys = list(groups)
    k = len(keys)
    if k < 2:
        return None, None, k

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        bx: list[float] = []
        by: list[float] = []
        for j in rng.integers(0, k, size=k):
            gx, gy = groups[keys[j]]
            bx.extend(gx)
            by.extend(gy)
        s = _spearman(bx, by)
        if s is not None:
            draws.append(s)
    # Too many degenerate resamples (constant vectors) means the interval is not
    # trustworthy; report undefined rather than a bound built from a few draws.
    if len(draws) < n_boot // 2:
        return None, None, k
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), k


def correlate(
    x: list[float],
    y: list[float],
    *,
    labels: list[Any] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    if len(x) != len(y):
        raise ValueError("x and y must align 1:1")
    if labels is not None and len(labels) != len(x):
        raise ValueError("labels must align 1:1 with x/y")
    n = len(x)
    base = {
        "n": n,
        "n_clusters": None,
        "pearson": None,
        "spearman": None,
        "spearman_lower_95": None,
        "spearman_upper_95": None,
    }
    if n < 3:
        return {**base, "note": "n<3"}
    if len(set(x)) == 1 or len(set(y)) == 1:
        # A near-constant score vector (e.g. faithfulness pinned at 5.0 by
        # deterministic quant verdicts) makes correlation undefined, not 0.
        return {**base, "note": "constant input"}
    from scipy import stats

    out = {
        **base,
        "pearson": float(stats.pearsonr(x, y).statistic),
        "spearman": float(stats.spearmanr(x, y).statistic),
    }
    if labels is not None:
        lo, hi, k = _cluster_bootstrap_ci(x, y, labels, n_boot=n_boot, seed=seed)
        out["n_clusters"] = k
        out["spearman_lower_95"] = lo
        out["spearman_upper_95"] = hi
    return out


def judge_decision(
    faithfulness: dict[str, Any],
    quality: dict[str, Any],
    *,
    faith_threshold: float = 0.7,
    quality_threshold: float = 0.6,
) -> dict[str, Any]:
    # Gate on the lower bound of the clustered CI, not the point estimate: a lucky
    # reshuffle of seven questions should not be able to pass the fine-tune.
    fl = faithfulness.get("spearman_lower_95")
    ql = quality.get("spearman_lower_95")
    use_local = fl is not None and ql is not None and fl > faith_threshold and ql > quality_threshold
    return {
        "use_local_for_iteration": use_local,
        "rule": (
            f"local judge for iteration if spearman_lower_95(faithfulness) > {faith_threshold} "
            f"AND spearman_lower_95(quality) > {quality_threshold}; otherwise Sonnet for all judging"
        ),
        "faithfulness_spearman": faithfulness.get("spearman"),
        "faithfulness_spearman_lower_95": fl,
        "faithfulness_n_clusters": faithfulness.get("n_clusters"),
        "quality_spearman": quality.get("spearman"),
        "quality_spearman_lower_95": ql,
        "quality_n_clusters": quality.get("n_clusters"),
    }
