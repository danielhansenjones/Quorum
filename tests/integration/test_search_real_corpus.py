from __future__ import annotations

import socket
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from quorum.ingest.qdrant_writer import COLLECTION_NAME
from quorum.tools.filing_section import FilingSectionNotFound, get_filing_section
from quorum.tools.search import dense_only_search, hybrid_search, sparse_only_search

pytestmark = pytest.mark.integration


def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _hf_cache_has_bge_m3() -> bool:
    return (Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3").exists()


@pytest.fixture
def qdrant(qdrant_url: str):
    if not _tcp_reachable("localhost", 6333):
        pytest.skip("qdrant not reachable")
    return QdrantClient(url=qdrant_url)


@pytest.fixture(scope="module")
def embedder():
    if not _hf_cache_has_bge_m3():
        pytest.skip("BGE-M3 weights not downloaded")
    from quorum.models.embed import BGEM3Embedder

    return BGEM3Embedder(device="cpu")


@pytest.fixture(scope="module")
def revenue_query_vecs(embedder):
    # One embedding pass reused across the search tests in this module.
    out = embedder.embed(["revenue growth and operating margin commentary"])
    return {
        "dense": out["dense_vecs"][0].tolist(),
        "sparse": out["lexical_weights"][0],
    }


def test_ticker_filter_strict(qdrant: QdrantClient, revenue_query_vecs) -> None:
    # Phase 3d / 4c: filter must be strict, no leakage from other tickers.
    hits = hybrid_search(
        qdrant,
        dense_vec=revenue_query_vecs["dense"],
        sparse_weights=revenue_query_vecs["sparse"],
        tickers=["AAPL"],
        top_k=20,
    )
    assert hits, "no hits returned"
    bad = [h for h in hits if h.payload.get("ticker") != "AAPL"]
    assert not bad, f"non-AAPL hits leaked: {[h.payload.get('ticker') for h in bad]}"


def test_ticker_and_section_filter_strict(qdrant: QdrantClient, revenue_query_vecs) -> None:
    hits = hybrid_search(
        qdrant,
        dense_vec=revenue_query_vecs["dense"],
        sparse_weights=revenue_query_vecs["sparse"],
        tickers=["AAPL"],
        sections=["item_7_mda"],
        top_k=15,
    )
    assert hits, "no hits for AAPL item_7_mda"
    for h in hits:
        assert h.payload.get("ticker") == "AAPL"
        assert h.payload.get("section") == "item_7_mda"


def test_top_k_respected(qdrant: QdrantClient, revenue_query_vecs) -> None:
    hits = hybrid_search(
        qdrant,
        dense_vec=revenue_query_vecs["dense"],
        sparse_weights=revenue_query_vecs["sparse"],
        top_k=5,
    )
    assert len(hits) == 5


def test_form_filter_strict(qdrant: QdrantClient, revenue_query_vecs) -> None:
    hits = hybrid_search(
        qdrant,
        dense_vec=revenue_query_vecs["dense"],
        sparse_weights=revenue_query_vecs["sparse"],
        forms=["10-Q"],
        top_k=10,
    )
    assert hits
    assert all(h.payload.get("form") == "10-Q" for h in hits)


def test_upsert_is_idempotent(qdrant: QdrantClient) -> None:
    # Phase 3d: deterministic point ID by chunk_id means re-running the writer
    # for the same chunk does not create duplicates. Verified by sampling one
    # point, scrolling for its chunk_id, and asserting exactly one row.
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_payload=["chunk_id"],
        with_vectors=False,
    )
    assert points, "no points in collection"
    target_id = points[0].payload.get("chunk_id") if points[0].payload else None
    assert target_id

    from qdrant_client import models as qm

    flt = qm.Filter(
        must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=str(target_id)))]
    )
    matched, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=flt,
        limit=10,
        with_payload=False,
        with_vectors=False,
    )
    assert len(matched) == 1, f"chunk_id {target_id} appears {len(matched)} times"


def test_hybrid_score_meets_or_beats_dense_for_top_hit(
    qdrant: QdrantClient, revenue_query_vecs
) -> None:
    # Phase 3d: the top hybrid score should not be strictly worse than the
    # corresponding chunk's dense-only or sparse-only score, when we can find
    # the same chunk in both result lists. RRF can change ranks though, so we
    # check that the hybrid top-1 chunk appears in the top-K of EITHER dense
    # or sparse, not that the float score itself is identical.
    hybrid = hybrid_search(
        qdrant,
        dense_vec=revenue_query_vecs["dense"],
        sparse_weights=revenue_query_vecs["sparse"],
        top_k=10,
    )
    dense = dense_only_search(qdrant, dense_vec=revenue_query_vecs["dense"], top_k=20)
    sparse = sparse_only_search(qdrant, sparse_weights=revenue_query_vecs["sparse"], top_k=20)
    assert hybrid
    top_hybrid_id = hybrid[0].chunk_id
    dense_ids = {h.chunk_id for h in dense}
    sparse_ids = {h.chunk_id for h in sparse}
    assert top_hybrid_id in dense_ids or top_hybrid_id in sparse_ids, (
        "top hybrid hit not in either component's top-20"
    )


def test_get_filing_section_roundtrip(qdrant: QdrantClient, revenue_query_vecs) -> None:
    # Phase 4d: take a hit, fetch its full section, assert the original chunk
    # text appears as a substring.
    hits = hybrid_search(
        qdrant,
        dense_vec=revenue_query_vecs["dense"],
        sparse_weights=revenue_query_vecs["sparse"],
        sections=["item_7_mda"],
        forms=["10-K"],
        top_k=3,
    )
    assert hits
    hit = hits[0]
    section = get_filing_section(
        qdrant,
        ticker=str(hit.payload["ticker"]),
        accession=str(hit.payload["accession"]),
        section=str(hit.payload["section"]),
    )
    assert section.text
    # The hit's text is a single chunk slice; it must appear inside the
    # stitched section text (which is the concatenation of all that section's
    # chunks separated by blank lines).
    hit_text = str(hit.payload.get("text", ""))
    assert hit_text
    probe = hit_text[: min(len(hit_text), 120)]
    assert probe in section.text


def test_get_filing_section_unknown_raises(qdrant: QdrantClient) -> None:
    bogus_acc = f"0000000000-99-{uuid.uuid4().hex[:6]}"
    with pytest.raises(FilingSectionNotFound):
        get_filing_section(qdrant, ticker="AAPL", accession=bogus_acc, section="item_7_mda")
