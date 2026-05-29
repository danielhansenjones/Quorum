# Walking skeleton (Phase 1.5, throwaway)

Single-file vertical slice. AAPL + MSFT, profitability axis, dense-only BGE-M3, Qdrant in-process, Sonnet analyst, FastAPI synchronous endpoint. No checkpointer, no LLM cache, no critic, no eval harness. Production phases reimplement everything; do not extend this code.

## What this proves

That EDGAR fetch, HTML parse, BGE-M3 embedding, Qdrant upsert/search, Anthropic Sonnet, and FastAPI all wire together end to end on this machine, before nine layered phases assume they do.

## Run

```
uv sync
cp .env.example .env  # then edit .env, set EDGAR_UA
./secret-run uv run python -m uvicorn spike.walking_skeleton:app
```

Requires [uv](https://docs.astral.sh/uv/).

- `EDGAR_UA` loads from `.env` via `python-dotenv` (SEC requires an identifying contact; see `.env.example`).
- `ANTHROPIC_API_KEY` comes from the keyring via `./secret-run` (libsecret / `secret-tool`). If you don't have secret-run set up, uncomment `ANTHROPIC_API_KEY` in your `.env` instead and drop the `./secret-run` prefix.
- `python -m uvicorn` (not bare `uvicorn`) so the `spike` package is importable without declaring a build backend for the throwaway spike.
- `.env` should be gitignored once you `git init`; never commit it.

First run downloads BGE-M3 weights (~2 GB) and the FlagEmbedding-pulled torch wheel (large). One-time cost. EDGAR responses and HTML are cached under `spike/_cache/` so subsequent runs only re-embed and re-call Sonnet.

Once `uvicorn` reports the ingest line for both tickers:

```
curl -X POST http://127.0.0.1:8000/compare \
  -H 'content-type: application/json' \
  -d '{"question":"Compare AAPL and MSFT on profitability over the most recent fiscal year."}'
```

You should get a JSON response with a `report` field containing prose plus inline `[TICKER:N]` citations, and an `evidence` field listing the chunk IDs and retrieval scores that backed each company's portion.

## Done criterion

The above curl returns prose with at least one verifiable citation, end to end, from a clean clone in under 30 minutes of setup (model download dominates).

## What's intentionally missing

- Item-level parsing (the whole 10-K is treated as one bag of chunks)
- XBRL facts and the concept-aliases path (numbers come from the LLM reading the text, which is exactly what the production design fixes)
- Postgres, checkpointer, trace events
- LLM cache, embedding cache
- Critic, multi-axis routing, classifier, refusal, re-plan
- Hybrid (sparse) retrieval
- Determinism guarantees (RRF tiebreaker, canonical-JSON cache key)
- Any tests

All of those are real Phase 1+ items. The spike's only job is to prove the wiring.

## Why a separate `spike/` directory

Throwaway code; keeping it out of `src/quorum/` avoids the temptation to grow it. Delete the directory once Phase 2+ produces the real ingest and tools.
