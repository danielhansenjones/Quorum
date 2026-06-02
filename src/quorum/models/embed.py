from __future__ import annotations

from typing import Any, TypedDict

from FlagEmbedding import BGEM3FlagModel

# Decision 9 (docs/decisions.md, ARCHITECTURE 6): BGE-M3 runs on CPU on the
# 9950x3d. Leaves the 16GB GPU for vLLM serving + KV cache. fp16 on CPU is
# slower than fp32 on x86, so use_fp16 flips to False when devices=cpu.
DEFAULT_DEVICE = "cpu"
DEFAULT_MODEL = "BAAI/bge-m3"
DENSE_DIM = 1024


class EmbedResult(TypedDict, total=False):
    dense_vecs: Any
    lexical_weights: list[dict[str, float]]


class BGEM3Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = DEFAULT_DEVICE,
        use_fp16: bool | None = None,
    ) -> None:
        if use_fp16 is None:
            use_fp16 = device != "cpu"
        self.model_name = model_name
        self.device = device
        # return_colbert_vecs hardcoded off (v1; ColBERT multi-vector is a v2
        # addition that requires a Qdrant collection recreation + re-ingest).
        self._model = BGEM3FlagModel(
            model_name,
            devices=device,
            use_fp16=use_fp16,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> EmbedResult:
        # ColBERT off regardless of caller wishes; that is the v1 contract.
        out = self._model.encode(
            texts,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )
        # The FlagEmbedding library returns numpy arrays for dense_vecs. The
        # callers serialize to list when crossing process boundaries; we keep
        # numpy here for cheap downstream math.
        return out  # type: ignore[no-any-return]


def health_check(embedder: BGEM3Embedder, probe: str = "Apple Inc reported revenue") -> None:
    out = embedder.embed([probe])
    dense = out["dense_vecs"]
    if dense.shape[-1] != DENSE_DIM:
        raise RuntimeError(f"BGE-M3 dense dim {dense.shape[-1]} != expected {DENSE_DIM}")
    sparse = out["lexical_weights"]
    if not sparse or not sparse[0]:
        raise RuntimeError("BGE-M3 sparse weights empty for finance probe")
