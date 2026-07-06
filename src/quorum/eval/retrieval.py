from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Two query populations with different notions of "good retrieval":
# - freeform: a specific answer exists; labeled positives are the chunks that
#   contain it. Positives are often near-duplicates across filings (10-K vs
#   10-Q repeats), so success@k (any positive in top k) is the headline and
#   plain recall@k is secondary.
# - planner: the fixed axis query is generic by construction, so there is no
#   single right chunk. Labels mark pooled candidates as substantive risk
#   content vs boilerplate/cross-reference stubs; the metric is judged
#   precision@k over the top k.
POPULATIONS = ("freeform", "planner")


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    id: str
    query: str
    ticker: str
    sections: list[str] | None
    population: str
    relevant_chunk_ids: list[str]
    judged_chunk_ids: list[str]
    gold_case: str | None = None
    notes: str = ""


def load_retrieval_dataset(path: Path) -> list[RetrievalQuery]:
    raw = yaml.safe_load(path.read_text()) or {}
    queries: list[RetrievalQuery] = []
    for q in raw.get("queries", []):
        population = str(q.get("population", "freeform"))
        if population not in POPULATIONS:
            raise ValueError(f"{q['id']}: unknown population {population!r}")
        queries.append(
            RetrievalQuery(
                id=str(q["id"]),
                query=str(q["query"]),
                ticker=str(q["ticker"]),
                sections=list(q["sections"]) if q.get("sections") else None,
                population=population,
                relevant_chunk_ids=list(q.get("relevant_chunk_ids") or []),
                judged_chunk_ids=list(q.get("judged_chunk_ids") or []),
                gold_case=q.get("gold_case"),
                notes=str(q.get("notes", "")),
            )
        )
    ids = [q.id for q in queries]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate query ids: {dupes}")
    return queries


def success_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall undefined without labeled positives")
    return sum(1 for cid in ranked[:k] if cid in relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def judged_precision_at_k(
    ranked: list[str], *, positive: set[str], judged: set[str], k: int
) -> dict[str, Any]:
    # Pooled labeling means a future index can surface chunks nobody judged;
    # count them separately instead of silently treating them as negatives.
    top = ranked[:k]
    judged_top = [cid for cid in top if cid in judged]
    unjudged = len(top) - len(judged_top)
    hits = sum(1 for cid in judged_top if cid in positive)
    precision = hits / len(judged_top) if judged_top else 0.0
    return {"precision": precision, "judged": len(judged_top), "unjudged": unjudged}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_rankings(
    queries: list[RetrievalQuery],
    rankings: dict[str, list[str]],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    planner_k: int = 5,
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    for q in queries:
        ranked = rankings[q.id]
        relevant = set(q.relevant_chunk_ids)
        row: dict[str, Any] = {
            "id": q.id,
            "population": q.population,
            "ticker": q.ticker,
            "n_relevant": len(relevant),
            "ranked": ranked,
        }
        if q.population == "freeform":
            row["success_at"] = {str(k): success_at_k(ranked, relevant, k) for k in ks}
            row["recall_at"] = {str(k): recall_at_k(ranked, relevant, k) for k in ks}
            row["mrr"] = reciprocal_rank(ranked, relevant)
            row["first_relevant_rank"] = next(
                (i for i, cid in enumerate(ranked, start=1) if cid in relevant), None
            )
        else:
            judged = set(q.judged_chunk_ids) | relevant
            row["judged_precision_at_k"] = judged_precision_at_k(
                ranked, positive=relevant, judged=judged, k=planner_k
            )
        per_query.append(row)

    freeform = [r for r in per_query if r["population"] == "freeform"]
    planner = [r for r in per_query if r["population"] == "planner"]
    summary: dict[str, Any] = {
        "n_queries": len(per_query),
        "n_freeform": len(freeform),
        "n_planner": len(planner),
    }
    if freeform:
        summary["freeform"] = {
            "success_at": {str(k): _mean([r["success_at"][str(k)] for r in freeform]) for k in ks},
            "recall_at": {str(k): _mean([r["recall_at"][str(k)] for r in freeform]) for k in ks},
            "mrr": _mean([r["mrr"] for r in freeform]),
        }
    if planner:
        summary["planner"] = {
            "judged_precision_at_k": _mean(
                [r["judged_precision_at_k"]["precision"] for r in planner]
            ),
            "planner_k": planner_k,
            "unjudged_total": sum(r["judged_precision_at_k"]["unjudged"] for r in planner),
        }
    return {"summary": summary, "per_query": per_query}
