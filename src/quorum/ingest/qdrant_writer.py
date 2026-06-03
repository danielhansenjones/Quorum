from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qm

# Collection contract (Phase 3d, ARCHITECTURE 4.1):
#   dense: 1024-d cosine. BGE-M3 dense_vecs.
#   sparse: BGE-M3 learned weights, NOT Qdrant's BM25. The distinction matters.
COLLECTION_NAME = "filings"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM = 1024

# Deterministic UUID namespace for chunk_id -> point id. Same chunk_id always
# maps to the same UUID, which is what makes upsert idempotent.
POINT_ID_NAMESPACE = uuid.UUID("11d4be3b-1c8a-4f6b-90a7-9c2e7f4af111")


@dataclass(frozen=True, slots=True)
class PointPayload:
    chunk_id: str
    ticker: str
    cik: str
    accession: str
    form: str
    section: str
    fiscal_period: str  # e.g. "FY2025", "Q1-2025"
    filing_date: str
    char_start: int
    char_end: int
    text: str


def point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                index=qm.SparseIndexParams(on_disk=False),
            ),
        },
    )
    # Payload indexes (Phase 3d): cheap exact-match filters at query time.
    for field in ("ticker", "section", "form", "fiscal_period"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )


def _to_sparse_vector(weights: dict[str, float]) -> qm.SparseVector:
    # BGE-M3 returns {token_id_str: weight}. Qdrant SparseVector wants parallel
    # indices/values arrays.
    indices = [int(k) for k in weights]
    values = list(weights.values())
    return qm.SparseVector(indices=indices, values=values)


def upsert_points(
    client: QdrantClient,
    *,
    payloads: Iterable[PointPayload],
    dense_vecs: list[list[float]],
    sparse_weights: list[dict[str, float]],
) -> int:
    payload_list = list(payloads)
    if not (len(payload_list) == len(dense_vecs) == len(sparse_weights)):
        raise ValueError(
            "payloads, dense_vecs, sparse_weights must have identical lengths "
            f"(got {len(payload_list)}, {len(dense_vecs)}, {len(sparse_weights)})"
        )
    points = [
        qm.PointStruct(
            id=point_id_for(p.chunk_id),
            vector={
                DENSE_VECTOR_NAME: dense,
                SPARSE_VECTOR_NAME: _to_sparse_vector(sparse),
            },
            payload={
                "chunk_id": p.chunk_id,
                "ticker": p.ticker,
                "cik": p.cik,
                "accession": p.accession,
                "form": p.form,
                "section": p.section,
                "fiscal_period": p.fiscal_period,
                "filing_date": p.filing_date,
                "char_start": p.char_start,
                "char_end": p.char_end,
                "text": p.text,
            },
        )
        for p, dense, sparse in zip(payload_list, dense_vecs, sparse_weights, strict=True)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)
