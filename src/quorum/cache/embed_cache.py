from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

from diskcache import Cache


def build_embed_cache_key(*, model_name: str, text: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"emb:{model_name}:{text_hash}"


def open_embed_cache(cache_dir: Path) -> Cache:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Cache(str(cache_dir))


def cached_embed_batch[T](
    cache: Cache,
    *,
    model_name: str,
    texts: Sequence[str],
    embed_fn: Callable[[Sequence[str]], list[T]],
) -> list[T]:
    # Per-text key so a partial batch overlap with prior calls still wins on the
    # known texts. Only the misses get sent to embed_fn, preserving original
    # input order on return.
    keys = [build_embed_cache_key(model_name=model_name, text=t) for t in texts]
    results: list[T | None] = [None] * len(texts)
    miss_indices: list[int] = []

    for i, k in enumerate(keys):
        hit = cache.get(k, default=_MISS)
        if hit is _MISS:
            miss_indices.append(i)
        else:
            results[i] = hit

    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        fresh = embed_fn(miss_texts)
        if len(fresh) != len(miss_indices):
            raise RuntimeError(
                f"embed_fn returned {len(fresh)} vectors for {len(miss_indices)} inputs"
            )
        for idx_into_misses, original_idx in enumerate(miss_indices):
            results[original_idx] = fresh[idx_into_misses]
            cache[keys[original_idx]] = fresh[idx_into_misses]

    if any(r is None for r in results):
        raise RuntimeError("embed cache produced None entries; this is a bug")
    return [r for r in results if r is not None]


class _MissSentinel:
    __slots__ = ()


_MISS = _MissSentinel()
