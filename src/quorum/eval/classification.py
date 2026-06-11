from __future__ import annotations

from typing import Any

from quorum.eval.runner import GoldCase
from quorum.graph.axis_config import SUPPORTED_AXES
from quorum.models.router import ChatClient


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def axis_metrics(predicted: list[set[str]], expected: list[set[str]]) -> dict[str, Any]:
    # Per-axis precision/recall/F1 plus exact-set-match over the whole set.
    # Deterministic given predictions; the LLM only produced the predictions.
    if len(predicted) != len(expected):
        raise ValueError("predicted and expected must align 1:1")
    per_axis: dict[str, dict[str, float]] = {}
    for axis in SUPPORTED_AXES:
        tp = sum(1 for p, e in zip(predicted, expected, strict=True) if axis in p and axis in e)
        fp = sum(1 for p, e in zip(predicted, expected, strict=True) if axis in p and axis not in e)
        fn = sum(1 for p, e in zip(predicted, expected, strict=True) if axis not in p and axis in e)
        per_axis[axis] = _prf(tp, fp, fn)
    exact = sum(1 for p, e in zip(predicted, expected, strict=True) if p == e)
    n = len(expected)
    macro_f1 = sum(per_axis[a]["f1"] for a in SUPPORTED_AXES) / len(SUPPORTED_AXES)
    return {
        "n": n,
        "per_axis": per_axis,
        "exact_set_match": exact / n if n else 0.0,
        "macro_f1": macro_f1,
    }


def refusal_metrics(predicted_refuse: list[bool], expected_refuse: list[bool]) -> dict[str, Any]:
    if len(predicted_refuse) != len(expected_refuse):
        raise ValueError("predicted and expected must align 1:1")
    tp = sum(1 for p, e in zip(predicted_refuse, expected_refuse, strict=True) if p and e)
    fp = sum(1 for p, e in zip(predicted_refuse, expected_refuse, strict=True) if p and not e)
    fn = sum(1 for p, e in zip(predicted_refuse, expected_refuse, strict=True) if not p and e)
    tn = sum(1 for p, e in zip(predicted_refuse, expected_refuse, strict=True) if not p and not e)
    out = _prf(tp, fp, fn)
    out["tn"] = tn
    out["accuracy"] = (tp + tn) / len(expected_refuse) if expected_refuse else 0.0
    return out


def _get_path(d: dict[str, Any], path: str) -> float:
    cur: Any = d
    for k in path.split("."):
        cur = cur[k]
    return float(cur)


def compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    # 10f regression gate: flag any guarded metric that dropped more than the
    # baseline's tolerance (default 0.03 = 3 points). Pure; the driver decides
    # what to do with a non-empty `regressions` list.
    tol = float(baseline.get("tolerance", 0.03))
    regressions: list[dict[str, Any]] = []
    for metric, base_val in baseline["metrics"].items():
        cur_val = _get_path(current, metric)
        drop = float(base_val) - cur_val
        if drop > tol:
            regressions.append(
                {"metric": metric, "baseline": float(base_val), "current": cur_val, "drop": drop}
            )
    return {"ok": not regressions, "tolerance": tol, "regressions": regressions}


def run_classification_eval(
    cases: list[GoldCase], *, classifier_client: ChatClient
) -> dict[str, Any]:
    # Drives classify + resolve over the gold set (classifier calls only; no
    # embedder or analyst), then scores axes and refusal deterministically.
    # Predicted refusal mirrors the graph's routing: out_of_scope OR fewer than
    # two in-corpus tickers resolved.
    from quorum.graph.nodes.classify import classify
    from quorum.graph.nodes.resolve import resolve

    pred_axes: list[set[str]] = []
    exp_axes: list[set[str]] = []
    pred_refuse: list[bool] = []
    exp_refuse: list[bool] = []
    for c in cases:
        out = classify(c.question, client=classifier_client)
        resolved = resolve(list(out.get("companies_raw") or []))
        refuse = bool(out.get("out_of_scope")) or len(resolved.get("tickers") or []) < 2
        pred_axes.append(set(out.get("axes") or []))
        exp_axes.append(set(c.expected_axes))
        pred_refuse.append(refuse)
        exp_refuse.append(c.expected_status == "refused")
    return {
        "n": len(cases),
        "axis": axis_metrics(pred_axes, exp_axes),
        "refusal": refusal_metrics(pred_refuse, exp_refuse),
    }
