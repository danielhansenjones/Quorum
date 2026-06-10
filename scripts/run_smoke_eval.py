from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from diskcache import Cache
from qdrant_client import QdrantClient

from quorum.config.settings import get_settings
from quorum.eval.runner import load_gold, run_all
from quorum.graph.build import build_graph
from quorum.models.embed import BGEM3Embedder
from quorum.models.router import get_client
from quorum.trace.logger import configure_logging, get_logger
from quorum.trace.writer import TraceWriter, open_pool

DEFAULT_GOLD = Path("eval/datasets/v1/gold.yaml")
DEFAULT_OUT_ROOT = Path("eval/runs")


def _build_embed_query(
    embedder: BGEM3Embedder,
) -> Callable[[str], tuple[list[float], dict[str, float]]]:
    def embed_query(text: str) -> tuple[list[float], dict[str, float]]:
        # One encode call returns both dense + sparse; graph callers want a tuple.
        out = embedder.embed([text])
        return out["dense_vecs"][0].tolist(), out["lexical_weights"][0]

    return embed_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Quorum smoke eval against the gold set")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--ids",
        type=str,
        default="",
        help="Comma-separated case IDs to run; empty = run all gold cases.",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Subdirectory name under out-root; defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Score faithfulness + quality with the Sonnet judge (extra API cost).",
    )
    parser.add_argument(
        "--critic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Critic node on/off (Phase 12a). Use --no-critic for the baseline arm.",
    )
    parser.add_argument(
        "--rebuttal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Critic<->analyst rebuttal loop on/off (Phase 13a). The +rebuttal arm.",
    )
    parser.add_argument(
        "--agentic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tiered agentic analyst (legwork loop + Sonnet write) on/off (Phase 13c). "
        "The +agentic arm; falls back to single-shot on failure.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Shared diskcache dir for the LLM cache (Phase 12f). Omit to disable caching.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write trace_events for cost accounting (Phase 12c). Needed for the A/B campaign.",
    )
    args = parser.parse_args()

    configure_logging()
    log = get_logger("smoke_eval")
    settings = get_settings()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("smoke_eval_missing_api_key")
        print("ANTHROPIC_API_KEY not set; run via ./secret-run", flush=True)
        return 2

    run_id = args.run_id or datetime.now(UTC).strftime("smoke-%Y%m%dT%H%M%SZ")
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = open_pool(
        conninfo=settings.postgres_url,
        min_size=settings.pg_pool_min,
        max_size=settings.pg_pool_max,
    )
    qdrant = QdrantClient(url=settings.qdrant_url)
    embedder = BGEM3Embedder(device="cpu")
    embed_query = _build_embed_query(embedder)

    classifier_client = get_client("classifier", vllm_url=settings.vllm_url)
    sonnet_client = get_client("analyst")
    legwork_client = get_client("legwork", vllm_url=settings.vllm_url) if args.agentic else None

    llm_cache = Cache(str(args.cache_dir)) if args.cache_dir else None
    trace = TraceWriter(pool) if args.trace else None

    log.info(
        "smoke_eval_start",
        run_id=run_id,
        classifier_backend=classifier_client.backend,
        classifier_model=classifier_client.model,
        analyst_model=sonnet_client.model,
        critic_enabled=args.critic,
        rebuttal_enabled=args.rebuttal,
        agentic_analyst=args.agentic,
        legwork_backend=legwork_client.backend if legwork_client else None,
        llm_cache=bool(args.cache_dir),
        trace=args.trace,
        gold=str(args.gold),
        out_dir=str(out_dir),
    )

    compiled = build_graph(
        classifier_client=classifier_client,
        sonnet_client=sonnet_client,
        pool=pool,
        qdrant=qdrant,
        embed_query=embed_query,
        critic_enabled=args.critic,
        rebuttal_enabled=args.rebuttal,
        agentic_analyst=args.agentic,
        legwork_client=legwork_client,
        llm_cache=llm_cache,
        trace=trace,
    )

    try:
        if args.ids:
            wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
            full_path = args.gold
            cases = [c for c in load_gold(full_path) if c.id in wanted]
            tmp_gold = out_dir / "_subset_gold.yaml"
            import yaml

            tmp_gold.write_text(
                yaml.safe_dump(
                    {
                        "cases": [
                            {
                                "id": c.id,
                                "question": c.question,
                                "expected_status": c.expected_status,
                                "expected_axes": c.expected_axes,
                                "expected_tickers": c.expected_tickers,
                                "notes": c.notes,
                            }
                            for c in cases
                        ]
                    }
                )
            )
            gold_to_run = tmp_gold
        else:
            gold_to_run = args.gold

        judge_kwargs = (
            # Judge calls share the run's LLM disk cache: a crashed or repeated
            # arm re-judges identical reports for free (once-only insurance).
            {"pool": pool, "qdrant": qdrant, "judge_client": sonnet_client, "llm_cache": llm_cache}
            if args.judge
            else {}
        )
        run_config = {
            "critic_enabled": args.critic,
            "rebuttal_enabled": args.rebuttal,
            "agentic_analyst": args.agentic,
            "llm_cache": bool(args.cache_dir),
            "trace": args.trace,
            "judge": args.judge,
        }
        summary = run_all(
            gold_to_run,
            compiled_graph=compiled,
            out_dir=out_dir,
            run_config=run_config,
            **judge_kwargs,
        )
    finally:
        pool.close()

    log.info("smoke_eval_complete", run_id=run_id, summary=summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
