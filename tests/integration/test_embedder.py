from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Loading BGE-M3 is heavy (~2GB weight load on first run, ~5s on warm CPU).
# Mark these as integration so the default unit run stays fast.
pytestmark = pytest.mark.integration


def _hf_cache_has_bge_m3() -> bool:
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3",
    ]
    return any(p.exists() for p in candidates)


@pytest.fixture(scope="module")
def embedder():
    from quorum.models.embed import BGEM3Embedder

    if not _hf_cache_has_bge_m3():
        pytest.skip(
            "BGE-M3 weights not downloaded; first run is ~2GB. "
            "Run ingest once (python -m quorum.ingest.run) to pull them."
        )
    return BGEM3Embedder(device="cpu")


def test_dense_dim_1024(embedder) -> None:
    out = embedder.embed(["Apple Inc reported revenue of 383 billion."])
    dense = out["dense_vecs"]
    assert dense.shape[-1] == 1024


def test_deterministic_dense(embedder) -> None:
    text = "Apple Inc reported revenue of 383 billion."
    a = embedder.embed([text])["dense_vecs"]
    b = embedder.embed([text])["dense_vecs"]
    # Strict equality on the same model + fp32 CPU path; ARCHITECTURE relies on
    # this for the determinism contract.
    assert np.array_equal(a, b)


def test_sparse_weights_non_empty(embedder) -> None:
    out = embedder.embed(["Apple Inc filed its 10-K annual report."])
    sparse = out["lexical_weights"]
    assert len(sparse) == 1
    assert isinstance(sparse[0], dict)
    assert len(sparse[0]) > 0, "BGE-M3 sparse weights should not be empty"


def test_batch_size_32(embedder) -> None:
    texts = [f"document number {i} about earnings and revenue" for i in range(32)]
    out = embedder.embed(texts)
    assert out["dense_vecs"].shape == (32, 1024)
    assert len(out["lexical_weights"]) == 32


def test_colbert_off(embedder) -> None:
    # v1 contract: ColBERT multi-vector is NOT computed. The encode call should
    # not return colbert_vecs even if the caller requests it.
    out = embedder.embed(["test"], return_dense=True, return_sparse=True)
    assert "colbert_vecs" not in out or not out.get("colbert_vecs")
