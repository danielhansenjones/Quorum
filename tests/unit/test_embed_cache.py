from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from quorum.cache.embed_cache import (
    build_embed_cache_key,
    cached_embed_batch,
    open_embed_cache,
)


def test_key_is_deterministic() -> None:
    k1 = build_embed_cache_key(model_name="bge-m3", text="hello world")
    k2 = build_embed_cache_key(model_name="bge-m3", text="hello world")
    assert k1 == k2


def test_different_model_distinct_key() -> None:
    k1 = build_embed_cache_key(model_name="bge-m3", text="x")
    k2 = build_embed_cache_key(model_name="bge-large", text="x")
    assert k1 != k2


def test_batch_misses_then_hits(tmp_cache_dir: Path) -> None:
    cache = open_embed_cache(tmp_cache_dir)
    calls: list[Sequence[str]] = []

    def embed_fn(texts: Sequence[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[1.0, 2.0]] * len(texts)

    out1 = cached_embed_batch(cache, model_name="bge-m3", texts=["a", "b", "c"], embed_fn=embed_fn)
    assert len(out1) == 3
    assert calls == [["a", "b", "c"]]

    out2 = cached_embed_batch(cache, model_name="bge-m3", texts=["a", "b", "c"], embed_fn=embed_fn)
    assert out2 == out1
    assert calls == [["a", "b", "c"]], "second call should hit cache entirely"


def test_partial_overlap_only_misses_get_embedded(tmp_cache_dir: Path) -> None:
    cache = open_embed_cache(tmp_cache_dir)
    seen_inputs: list[list[str]] = []

    def embed_fn(texts: Sequence[str]) -> list[list[float]]:
        seen_inputs.append(list(texts))
        return [[float(len(t))] for t in texts]

    cached_embed_batch(cache, model_name="bge-m3", texts=["a", "b"], embed_fn=embed_fn)
    cached_embed_batch(cache, model_name="bge-m3", texts=["b", "c", "a"], embed_fn=embed_fn)
    assert seen_inputs == [["a", "b"], ["c"]]
