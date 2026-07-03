# Quorum

[![CI](https://github.com/danielhansenjones/Quorum/actions/workflows/ci.yml/badge.svg)](https://github.com/danielhansenjones/Quorum/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![status-match](https://img.shields.io/badge/status--match-35%2F41-success)](#results)
[![faithfulness](https://img.shields.io/badge/faithfulness-4.5%2F5-success)](#results)
[![quality](https://img.shields.io/badge/quality-4.6%2F5-success)](#results)

[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-hybrid%20search-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Postgres](https://img.shields.io/badge/Postgres-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Sonnet%20%2B%20Haiku-D97757?logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)

**Multi-agent financial research over SEC filings.** A quorum of per-axis analysts compares public companies on profitability, growth, leverage, and risk; a synthesizer reconciles them; an agentic critic fact-checks every claim against the source data before the report is written.

Ask `"Compare AAPL and MSFT on profitability and growth"` and Quorum classifies the request, resolves the tickers, fans out one analyst per axis in parallel, fact-checks the draft with a tool-using critic, and returns a cited markdown report. Quant claims resolve to XBRL facts; qualitative claims resolve to passages from the actual 10-K and 10-Q filings. Every claim the critic flags is dropped, softened, or counter-cited.

## Highlights

- **A graph, not a chain.** Analysts fan out in parallel (LangGraph `Send`), an `assess` node re-plans only the weakly-grounded axes within a step budget, and an agentic critic verifies the draft before synthesis. Branching, re-planning, and a bounded agent loop in one graph.
- **Agentic fact-checking.** The critic runs a bounded tool loop (5 turns / 90s) over the same XBRL facts and filing text the analysts used, flags unsupported claims, and the synthesizer acts on every flag.
- **Grounded in real data.** 12 companies across Big Tech, Consumer Staples, and Pharma; the latest 10-K plus four 10-Qs each (~60 filings, ~6000 chunks). Quant -> XBRL facts in Postgres; qual -> hybrid search over Qdrant (BGE-M3 dense + learned sparse).
- **Measured, not asserted.** A 41-case gold set scored by an LLM-as-judge harness: status-match 35/41, faithfulness 4.51/5, quality 4.57/5 (0 judge failures). Classifier axis macro-F1 0.92; refusal precision and recall 1.0.
- **Honest eval methodology.** A judge-correlation study tested a cheap local 7B judge against Sonnet and rejected it (quality Spearman 0.11). Sonnet judges everything; the decision and its gates are checked into [`eval/judge_config.yaml`](eval/judge_config.yaml).
- **Does the critic earn its cost?** A paired A/B harness (critic on vs off) with bootstrap confidence intervals and an incorporation metric is wired to quantify the critic's effect on faithfulness, quality, and dollars. The same harness carries two further default-off arms: a critic-analyst rebuttal loop and a tiered agentic analyst (cheap legwork tool-loop, Sonnet write).
- **Durable by construction.** A Postgres checkpointer writes state at every super-step; `/runs/{id}/resume` re-drives from the last checkpoint, and re-run nodes hit a canonical-JSON LLM cache, so they fire zero new API calls when inputs are unchanged.
- **Real cost accounting.** Every LLM call writes a trace row with token counts and a dollar figure. A full multi-axis comparison runs about $0.10, and the critic is the measured cost driver.
- **Two surfaces.** FastAPI with an SSE stream of node events (watch the agent work), and an MCP server exposing the six tools plus a high-level `compare_companies` for Claude Desktop.
- **Typed and gated.** Pydantic v2 state with a discriminated citation union and parallel-write-safe reducers; mypy strict on the typed core; ruff lint and format; CI on every push.

## Demo

> **Watch it run.** A terminal recording goes here. Until then, [`scripts/demo.py`](scripts/demo.py) renders the full agent trajectory live over the SSE stream; representative output is below.

<!--
Drop the recording in here once captured:
  asciinema:  [![demo](https://asciinema.org/a/<ID>.svg)](https://asciinema.org/a/<ID>)
  or a GIF:   ![Quorum demo](assets/demo.gif)   # commit it under a tracked dir, not an ignored one
Capture: warm the cache once (free, deterministic), then
  uv run python scripts/demo.py "Compare AAPL and MSFT on profitability and growth" --step 0.5 --cost
-->

```
# terminal 1 - start the API (Anthropic key is injected by ./secret-run, never printed)
./secret-run uv run uvicorn quorum.api.main:app --port 8000

# terminal 2 - stream a comparison and watch the agent work
uv run python scripts/demo.py "Compare AAPL and MSFT on profitability and growth" --step 0.5 --cost
```

Representative output (colorized in a real terminal):

```
  quorum > Compare AAPL and MSFT on profitability and growth

  classify     in-scope  axes=[profitability, growth]  mentions=[AAPL, MSFT]
  resolve      tickers=[AAPL, MSFT]
  plan         2 analyst task(s) -> fan-out: [profitability, growth]   budget=4
  analyze      profitability    done  grounding=ok  6 cites
  analyze      growth           done  grounding=ok  5 cites
  assess       2 axes  0 weak
  critic       agent loop  turns=2  status=ok  (4218 ms)
    [1] get_financial_concept(ticker=AAPL concept=NetIncomeLoss)  -> {"value":"93736000000","unit":"USD",...  [ok]
    [2] get_financial_concept(ticker=MSFT concept=NetIncomeLoss)  -> {"value":"88136000000","unit":"USD",...  [ok]
    [3] search_filings(query=cloud revenue growth drivers)        -> 4 hits across MSFT 10-Q/10-K           [ok]
    FLAG [growth] unsupported: MSFT Intelligent Cloud grew 31% year over year
         reason: filing states 'server products and cloud services revenue increased 21%', not 31%
  synthesize   status=ok  11 citations
  ------------------------------------------------------------------------
  REPORT       status=ok  11 citations  request_id=8f3c1d2a

    ## Profitability
    Apple's FY2024 net income of $93.7B exceeds Microsoft's $88.1B [AAPL:Q0] [MSFT:Q1] ...

    ## Growth
    Microsoft's cloud revenue grew 21% year over year [MSFT:Q2] (the 31% figure
    was dropped after the critic flagged it as unsupported) ...
  ------------------------------------------------------------------------
  COST         total $0.0972   cache_read=0%
    llm:analyst              $0.0361   in=8421  out=1206
    llm:critic               $0.0456   in=10220 out=804
    llm:synthesizer          $0.0151   in=6004  out=988
    llm:classifier           $0.0004   in=512   out=44
```

## How it works

```
question
  |
  v
[classify] --out of scope / no axis--------> [refuse] --> END
  |
  v
[resolve]  --fewer than 2 in-corpus tickers-> [refuse] --> END
  |
  v
[plan] <-------------------------------------+
  |  Send(axis) x N  (parallel fan-out)       |  re-plan only the
  v                                           |  weak axes, within budget
[analyze_axis]  ...  [analyze_axis]           |
  \________________|________________/         |
                   v                           |
              [assess]  --any axis weak?-------+
                   |  all grounded / budget spent
                   v
              [critic]   agentic: re-checks claims with the same tools (5-turn / 90s cap)
                   |
                   v
              [synthesize]   drops / softens / counter-cites every flagged claim
                   |
                   v
                  END
```

Four Docker Compose services back it:

| Service    | Role |
|------------|------|
| `postgres` (16)      | XBRL facts, LangGraph checkpointer, `trace_events` |
| `qdrant`             | hybrid vector index (BGE-M3 dense + learned sparse) |
| `vllm` (gpu profile) | Qwen 2.5 7B Instruct AWQ-4bit, the local classifier (optional) |
| `api` (api profile)  | FastAPI with the SSE stream of node events |

Anthropic Sonnet powers the analyst, synthesizer, critic, and canonical judge. Haiku is the classifier fallback when `VLLM_URL` is unset, so the system runs with no GPU. Full node-by-node behavior, the state schema, and the retrieval design are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Results

41-case gold set, judged by Sonnet ([`eval/judge_config.yaml`](eval/judge_config.yaml)):

| Metric | Value |
|--------|-------|
| Status match (ok / partial / refused) | 35 / 41 (0.85) |
| Faithfulness mean (32 answered cases) | 4.51 / 5 |
| Quality mean (32 answered reports)    | 4.57 / 5 |
| Quality-judge failures                | 0 |

Classifier, deterministic scoring over the full gold set:

| Metric | Value |
|--------|-------|
| Axis macro-F1                         | 0.92 |
| Axis exact-set-match                  | 0.85 |
| Refusal recall / precision / accuracy | 1.00 / 1.00 / 1.00 |

Faithfulness is deterministic for quant citations (value + unit + period checked against Postgres) and LLM-judged for qual citations. The judge-correlation study rejected a cheap local 7B judge on quality (Spearman 0.11 against Sonnet) and kept Sonnet as the sole judge. The critic-on-vs-off A/B harness (paired bootstrap CIs, incorporation metric) is wired and ready to run. Per-case artifacts behind these numbers (reports, scores, citations) are committed under [`eval/results/judged-full-v1-final/`](eval/results/judged-full-v1-final/). Full tables, the cost breakdown, and the honest analysis behind each number are in [ARCHITECTURE.md](ARCHITECTURE.md#eval-harness).

## Quickstart

```
cp .env.example .env   # set EDGAR_UA; SEC fair access requires a contact user agent
docker compose up -d postgres qdrant
uv sync --extra eval --group dev
uv run python -m quorum.ingest.run
```

Ingest pulls 12 companyfacts JSONs plus ~60 filings and embeds ~6000 chunks on CPU. Budget about an hour on a 9950X3D.

LLM calls need `ANTHROPIC_API_KEY` in the environment. `./secret-run` in the commands on this page is a local keyring wrapper (libsecret exec) that injects the key without echoing it anywhere; it is not part of the repo. `export ANTHROPIC_API_KEY=...` (or the commented line in `.env`) works in its place.

Then run the API and the demo from [Demo](#demo) above. `GET /ready` reports backing-service health:

```
curl -s localhost:8000/ready
{"ok":true,"checks":{"postgres":true,"qdrant":true}}
```

## API and MCP surface

| Endpoint | Behavior |
|----------|----------|
| `POST /compare`               | streams node events over SSE, then a final cited report |
| `GET /runs/{request_id}/resume` | re-drives an interrupted run from the last Postgres checkpoint |
| `GET /ready`                  | backing-service health (`postgres`, `qdrant`) |
| `GET /health`                 | liveness |

The same capability is exposed over MCP: the six low-level tools (`resolve_company`, `get_financial_concept`, `search_filings`, `get_filing_section`, `list_corpus`, `list_filings`) plus the high-level `compare_companies`, usable from Claude Desktop or the MCP inspector.

## Eval and cost

```
# run the 41-case gold set through the graph, write per-case JSON + summary
./secret-run uv run python scripts/run_smoke_eval.py

# add LLM-as-judge scoring (quant faithfulness + qual faithfulness + quality rubric)
./secret-run uv run python scripts/run_smoke_eval.py --judge

# per-request and per-node dollar cost from the trace rows
./secret-run uv run python scripts/run_cost_report.py

# paired A/B of two run dirs (e.g. baseline vs +critic) with bootstrap CIs
uv run python scripts/run_ab_compare.py eval/runs/baseline eval/runs/+critic --cost
```

A full multi-axis comparison runs about $0.10; refusals short-circuit to ~$0.001. The critic is the cost driver (~$0.023/turn). See [ARCHITECTURE.md](ARCHITECTURE.md#cost) for the per-node breakdown and the caching story.

## Engineering

- **Types as a gate.** Pydantic v2 state with a discriminated `Citation` union and reducers that are safe under the parallel fan-out. mypy strict on the typed core (`state`, `graph`, `tools`, `models`, `cache`).
- **CI on every push.** ruff lint, ruff format check, mypy, and the unit + smoke suites ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Integration, GPU, and token-billing eval suites are deliberately out of CI.
- **Resume that cannot silently rot.** The checkpointer uses an explicit msgpack allowlist; a completeness test fails the build if a state model is added without registering it, and another proves the allowlist blocks stray types on deserialize.
- **pre-commit** mirrors the CI lint and format gates.

## Limitations

A few honest edges, with the full list in [ARCHITECTURE.md](ARCHITECTURE.md#limitations):

- The corpus is fixed (latest 10-K + four 10-Qs per company). A question whose scope exceeds that window, or asks for a breakout the XBRL facts do not isolate, is answered on the available slice; flagging that under-scope and downgrading to `partial` is a v2 item.
- The `assess` node over-flags a few well-grounded qualitative axes as weak; the grounding heuristic is tuned for quant-fact density and under-credits qual evidence.
- The local Qwen classifier is a portfolio statement, not a v1 cost win: Haiku is cheaper than the GPU time at this scale. The honest framing is in the architecture doc.

## More

- [ARCHITECTURE.md](ARCHITECTURE.md) - node-by-node graph, state schema, retrieval, the full eval harness, cost, and limitations.
- [`scripts/demo.py`](scripts/demo.py) - the live trajectory renderer used in the demo above.
