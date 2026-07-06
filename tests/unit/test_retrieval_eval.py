from __future__ import annotations

from pathlib import Path

import pytest

from quorum.eval.retrieval import (
    evaluate_rankings,
    judged_precision_at_k,
    load_retrieval_dataset,
    recall_at_k,
    reciprocal_rank,
    success_at_k,
)

_RANKED = ["c1", "c2", "c3", "c4", "c5"]


def test_success_at_k() -> None:
    assert success_at_k(_RANKED, {"c3"}, 3) == 1.0
    assert success_at_k(_RANKED, {"c3"}, 2) == 0.0
    assert success_at_k(_RANKED, {"missing"}, 5) == 0.0
    assert success_at_k([], {"c1"}, 5) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(_RANKED, {"c1", "c4"}, 5) == 1.0
    assert recall_at_k(_RANKED, {"c1", "c4"}, 3) == 0.5
    assert recall_at_k(_RANKED, {"c1", "missing"}, 5) == 0.5


def test_recall_requires_positives() -> None:
    with pytest.raises(ValueError):
        recall_at_k(_RANKED, set(), 5)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(_RANKED, {"c1"}) == 1.0
    assert reciprocal_rank(_RANKED, {"c3", "c5"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(_RANKED, {"missing"}) == 0.0


def test_judged_precision_counts_unjudged_separately() -> None:
    out = judged_precision_at_k(_RANKED, positive={"c1", "c3"}, judged={"c1", "c2", "c3"}, k=5)
    # c4/c5 were never pooled, so they are unjudged, not negatives.
    assert out == {"precision": pytest.approx(2 / 3), "judged": 3, "unjudged": 2}


def test_judged_precision_empty_topk() -> None:
    out = judged_precision_at_k([], positive={"c1"}, judged={"c1"}, k=5)
    assert out["precision"] == 0.0
    assert out["judged"] == 0


def _write_dataset(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_load_dataset_roundtrip(tmp_path: Path) -> None:
    p = _write_dataset(
        tmp_path / "ds.yaml",
        """
queries:
  - id: q1
    query: supply chain concentration
    ticker: AAPL
    sections: [item_1a_risk_factors]
    population: freeform
    relevant_chunk_ids: [a, b]
    gold_case: happy_tech_risks
  - id: q2
    query: primary business risks
    ticker: PFE
    population: planner
    judged_chunk_ids: [c]
""",
    )
    queries = load_retrieval_dataset(p)
    assert [q.id for q in queries] == ["q1", "q2"]
    assert queries[0].sections == ["item_1a_risk_factors"]
    assert queries[0].relevant_chunk_ids == ["a", "b"]
    assert queries[1].sections is None
    assert queries[1].population == "planner"


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    p = _write_dataset(
        tmp_path / "ds.yaml",
        """
queries:
  - {id: q1, query: x, ticker: AAPL}
  - {id: q1, query: y, ticker: MSFT}
""",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_retrieval_dataset(p)


def test_load_dataset_rejects_unknown_population(tmp_path: Path) -> None:
    p = _write_dataset(
        tmp_path / "ds.yaml",
        "queries:\n  - {id: q1, query: x, ticker: AAPL, population: nonsense}\n",
    )
    with pytest.raises(ValueError, match="population"):
        load_retrieval_dataset(p)


def test_evaluate_rankings_mixed_populations(tmp_path: Path) -> None:
    p = _write_dataset(
        tmp_path / "ds.yaml",
        """
queries:
  - id: f1
    query: x
    ticker: AAPL
    population: freeform
    relevant_chunk_ids: [a]
  - id: f2
    query: y
    ticker: KO
    population: freeform
    relevant_chunk_ids: [z]
  - id: p1
    query: generic risks
    ticker: PFE
    population: planner
    relevant_chunk_ids: [g1]
    judged_chunk_ids: [g1, g2]
""",
    )
    queries = load_retrieval_dataset(p)
    rankings = {
        "f1": ["a", "b", "c"],
        "f2": ["b", "c", "z"],
        "p1": ["g1", "g2", "unpooled"],
    }
    out = evaluate_rankings(queries, rankings, ks=(1, 3), planner_k=3)
    s = out["summary"]
    assert s["n_freeform"] == 2 and s["n_planner"] == 1
    assert s["freeform"]["success_at"]["1"] == 0.5
    assert s["freeform"]["success_at"]["3"] == 1.0
    assert s["freeform"]["mrr"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert s["planner"]["judged_precision_at_k"] == pytest.approx(0.5)
    assert s["planner"]["unjudged_total"] == 1
    ranks = {r["id"]: r for r in out["per_query"]}
    assert ranks["f1"]["first_relevant_rank"] == 1
    assert ranks["f2"]["first_relevant_rank"] == 3
