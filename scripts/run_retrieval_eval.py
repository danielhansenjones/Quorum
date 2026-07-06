from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from quorum.config.settings import get_settings
from quorum.eval.retrieval import RetrievalQuery, evaluate_rankings, load_retrieval_dataset
from quorum.models.embed import BGEM3Embedder
from quorum.tools.search import dense_only_search, hybrid_search, sparse_only_search

DEFAULT_DATASET = Path("eval/datasets/retrieval_v1.yaml")
DEFAULT_OUT = Path("eval/results/retrieval-v1")
ARMS = ("hybrid", "dense", "sparse")


def _run_arm(
    arm: str,
    client: QdrantClient,
    q: RetrievalQuery,
    *,
    dense_vec: list[float],
    sparse_weights: dict[str, float],
    top_k: int,
) -> list[str]:
    if arm == "hybrid":
        hits = hybrid_search(
            client,
            dense_vec=dense_vec,
            sparse_weights=sparse_weights,
            tickers=[q.ticker],
            sections=q.sections,
            top_k=top_k,
        )
    elif arm == "dense":
        hits = dense_only_search(
            client, dense_vec=dense_vec, tickers=[q.ticker], sections=q.sections, top_k=top_k
        )
    elif arm == "sparse":
        hits = sparse_only_search(
            client,
            sparse_weights=sparse_weights,
            tickers=[q.ticker],
            sections=q.sections,
            top_k=top_k,
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return [h.chunk_id for h in hits]


def _dump_pool(
    client: QdrantClient,
    queries: list[RetrievalQuery],
    rankings_by_arm: dict[str, dict[str, list[str]]],
    out: Path,
) -> None:
    # Union of all arms' top-k per query, with payload text, so labeling sees
    # every candidate any compared system can surface (TREC-style pooling).
    # Fetch by deterministic point id (UUIDv5 of chunk_id) - no payload index
    # on chunk_id exists or is needed.
    from quorum.ingest.qdrant_writer import COLLECTION_NAME, point_id_for

    with out.open("w") as f:
        for q in queries:
            pool: list[str] = []
            for arm in rankings_by_arm:
                for cid in rankings_by_arm[arm][q.id]:
                    if cid not in pool:
                        pool.append(cid)
            points = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[point_id_for(cid) for cid in pool],
                with_payload=True,
            )
            by_id = {str((p.payload or {}).get("chunk_id")): p.payload or {} for p in points}
            candidates = [
                {
                    "chunk_id": cid,
                    "section": by_id.get(cid, {}).get("section"),
                    "form": by_id.get(cid, {}).get("form"),
                    "fiscal_period": by_id.get(cid, {}).get("fiscal_period"),
                    "text": by_id.get(cid, {}).get("text"),
                }
                for cid in pool
            ]
            f.write(
                json.dumps(
                    {
                        "query_id": q.id,
                        "query": q.query,
                        "ticker": q.ticker,
                        "sections": q.sections,
                        "population": q.population,
                        "candidates": candidates,
                    }
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quorum retrieval eval (hybrid vs dense vs sparse)"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--dump-pool",
        type=Path,
        default=None,
        help="Write pooled candidates (union of all arms' top-k, with text) to this JSONL and exit.",
    )
    args = parser.parse_args()

    queries = load_retrieval_dataset(args.dataset)
    if not args.dump_pool:
        unlabeled = [
            q.id for q in queries if q.population == "freeform" and not q.relevant_chunk_ids
        ]
        if unlabeled:
            print(f"unlabeled freeform queries: {unlabeled}", file=sys.stderr)
            return 2

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)
    embedder = BGEM3Embedder()
    embedded = embedder.embed([q.query for q in queries])

    rankings_by_arm: dict[str, dict[str, list[str]]] = {arm: {} for arm in ARMS}
    for i, q in enumerate(queries):
        dense_vec = embedded["dense_vecs"][i].tolist()
        sparse_weights = embedded["lexical_weights"][i]
        for arm in ARMS:
            rankings_by_arm[arm][q.id] = _run_arm(
                arm, client, q, dense_vec=dense_vec, sparse_weights=sparse_weights, top_k=args.top_k
            )

    if args.dump_pool:
        args.dump_pool.parent.mkdir(parents=True, exist_ok=True)
        _dump_pool(client, queries, rankings_by_arm, args.dump_pool)
        print(f"pool written to {args.dump_pool}")
        return 0

    from quorum.eval.runner import _git_provenance

    results = {arm: evaluate_rankings(queries, rankings_by_arm[arm]) for arm in ARMS}
    n_pairs = sum(len(q.relevant_chunk_ids) for q in queries)
    summary = {
        "dataset": str(args.dataset),
        "n_queries": len(queries),
        "n_labeled_pairs": n_pairs,
        "arms": {arm: results[arm]["summary"] for arm in ARMS},
        "run_config": {
            "embed_model": "BAAI/bge-m3",
            "top_k": args.top_k,
            "arms": list(ARMS),
        },
        "provenance": {
            **_git_provenance(),
            "started_at": datetime.now(UTC).isoformat(),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    per_query = {arm: results[arm]["per_query"] for arm in ARMS}
    (args.out / "per_query.json").write_text(json.dumps(per_query, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
