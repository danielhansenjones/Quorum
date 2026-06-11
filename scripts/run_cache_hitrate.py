from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient

from quorum.cache.llm_cache import open_llm_cache
from quorum.config.settings import get_settings
from quorum.eval.runner import load_gold, run_case
from quorum.graph.build import build_graph
from quorum.models.embed import BGEM3Embedder
from quorum.models.router import get_client
from quorum.trace.writer import open_pool

DEFAULT_GOLD = Path("eval/datasets/v1/gold.yaml")
DEFAULT_IDS = "happy_aapl_msft_profitability,happy_jnj_pfe_leverage,happy_googl_meta_growth"


def _embed_query(
    embedder: BGEM3Embedder,
) -> Callable[[str], tuple[list[float], dict[str, float]]]:
    def f(text: str) -> tuple[list[float], dict[str, float]]:
        out = embedder.embed([text])
        return out["dense_vecs"][0].tolist(), out["lexical_weights"][0]

    return f


def main() -> int:
    ap = argparse.ArgumentParser(
        description="10j: local LLM cache hit rate on a 2-pass unchanged eval set"
    )
    ap.add_argument("--ids", default=DEFAULT_IDS)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Run via ./secret-run (pass 1 makes real Sonnet calls).", flush=True)
        return 2

    settings = get_settings()
    pool = open_pool(conninfo=settings.postgres_url, min_size=1, max_size=settings.pg_pool_max)
    qdrant = QdrantClient(url=settings.qdrant_url)
    embedder = BGEM3Embedder(device="cpu")
    classifier = get_client("classifier", vllm_url=settings.vllm_url)
    sonnet = get_client("analyst")

    cache_dir = Path(tempfile.mkdtemp(prefix="quorum-cache-10j-"))
    cache = open_llm_cache(cache_dir)
    cache.stats(enable=True)

    compiled = build_graph(
        classifier_client=classifier,
        sonnet_client=sonnet,
        pool=pool,
        qdrant=qdrant,
        embed_query=_embed_query(embedder),
        llm_cache=cache,
    )

    wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
    cases = [c for c in load_gold(args.gold) if c.id in wanted]

    for c in cases:  # pass 1, cold: populates the cache
        run_case(c, compiled_graph=compiled)
    cache.stats(reset=True)  # discard pass-1 counts; keep tracking enabled

    for c in cases:  # pass 2, warm: identical inputs must hit
        run_case(c, compiled_graph=compiled)
    hits, misses = cache.stats()
    total = hits + misses
    rate = hits / total if total else 0.0

    print(
        json.dumps(
            {
                "ids": sorted(wanted),
                "pass2_hits": hits,
                "pass2_misses": misses,
                "pass2_total": total,
                "hit_rate": rate,
                "threshold": args.threshold,
            },
            indent=2,
        ),
        flush=True,
    )
    pool.close()
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
