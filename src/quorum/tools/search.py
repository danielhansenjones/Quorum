from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from quorum.ingest.qdrant_writer import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    _to_sparse_vector,
)


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]
    # Component scores when available; helpful for debugging fusion behavior.
    dense_score: float | None = None
    sparse_score: float | None = None


def _build_filter(
    *,
    tickers: list[str] | None,
    sections: list[str] | None,
    forms: list[str] | None,
    fiscal_periods: list[str] | None,
) -> qm.Filter | None:
    # Sequence[FieldCondition] (covariant) keeps mypy happy under the qdrant
    # client's invariant `list[Condition]` parameter.
    must: list[qm.FieldCondition] = []
    if tickers:
        must.append(qm.FieldCondition(key="ticker", match=qm.MatchAny(any=list(tickers))))
    if sections:
        must.append(qm.FieldCondition(key="section", match=qm.MatchAny(any=list(sections))))
    if forms:
        must.append(qm.FieldCondition(key="form", match=qm.MatchAny(any=list(forms))))
    if fiscal_periods:
        must.append(
            qm.FieldCondition(key="fiscal_period", match=qm.MatchAny(any=list(fiscal_periods)))
        )
    if not must:
        return None
    widened: Sequence[Any] = must
    return qm.Filter(must=list(widened))


def hybrid_search(
    client: QdrantClient,
    *,
    dense_vec: list[float],
    sparse_weights: dict[str, float],
    tickers: list[str] | None = None,
    sections: list[str] | None = None,
    forms: list[str] | None = None,
    fiscal_periods: list[str] | None = None,
    top_k: int = 10,
    prefetch_multiplier: int = 4,
) -> list[SearchHit]:
    # RRF over a dense prefetch + a sparse prefetch. Deterministic tiebreaker
    # on chunk_id ascending applied after fusion (ARCHITECTURE section 9).
    flt = _build_filter(
        tickers=tickers, sections=sections, forms=forms, fiscal_periods=fiscal_periods
    )
    prefetch_limit = top_k * prefetch_multiplier
    resp = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            qm.Prefetch(
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=flt,
            ),
            qm.Prefetch(
                query=_to_sparse_vector(sparse_weights),
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=flt,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    hits = [
        SearchHit(
            chunk_id=str(p.payload.get("chunk_id", p.id)) if p.payload else str(p.id),
            score=float(p.score),
            payload=dict(p.payload or {}),
        )
        for p in resp.points
    ]
    # Deterministic tiebreaker: when two hits have the same RRF score (rare but
    # possible at small prefetch_limit), sort ascending by chunk_id.
    hits.sort(key=lambda h: (-h.score, h.chunk_id))
    return hits


def dense_only_search(
    client: QdrantClient,
    *,
    dense_vec: list[float],
    tickers: list[str] | None = None,
    sections: list[str] | None = None,
    forms: list[str] | None = None,
    fiscal_periods: list[str] | None = None,
    top_k: int = 10,
) -> list[SearchHit]:
    flt = _build_filter(
        tickers=tickers, sections=sections, forms=forms, fiscal_periods=fiscal_periods
    )
    resp = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vec,
        using=DENSE_VECTOR_NAME,
        query_filter=flt,
        limit=top_k,
        with_payload=True,
    )
    hits = [
        SearchHit(
            chunk_id=str(p.payload.get("chunk_id", p.id)) if p.payload else str(p.id),
            score=float(p.score),
            payload=dict(p.payload or {}),
        )
        for p in resp.points
    ]
    hits.sort(key=lambda h: (-h.score, h.chunk_id))
    return hits


def sparse_only_search(
    client: QdrantClient,
    *,
    sparse_weights: dict[str, float],
    tickers: list[str] | None = None,
    sections: list[str] | None = None,
    forms: list[str] | None = None,
    fiscal_periods: list[str] | None = None,
    top_k: int = 10,
) -> list[SearchHit]:
    flt = _build_filter(
        tickers=tickers, sections=sections, forms=forms, fiscal_periods=fiscal_periods
    )
    resp = client.query_points(
        collection_name=COLLECTION_NAME,
        query=_to_sparse_vector(sparse_weights),
        using=SPARSE_VECTOR_NAME,
        query_filter=flt,
        limit=top_k,
        with_payload=True,
    )
    hits = [
        SearchHit(
            chunk_id=str(p.payload.get("chunk_id", p.id)) if p.payload else str(p.id),
            score=float(p.score),
            payload=dict(p.payload or {}),
        )
        for p in resp.points
    ]
    hits.sort(key=lambda h: (-h.score, h.chunk_id))
    return hits
