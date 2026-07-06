# Quorum

[![CI](https://github.com/danielhansenjones/Quorum/actions/workflows/ci.yml/badge.svg)](https://github.com/danielhansenjones/Quorum/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![refusal-accuracy](https://img.shields.io/badge/refusal--accuracy-9%2F9-success)](#results)
[![status-match](https://img.shields.io/badge/status--match-29%2F41-success)](#results)
[![faithfulness](https://img.shields.io/badge/faithfulness-4.6%2F5-success)](#results)
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
- **Measured, not asserted.** A 41-case gold set scored by an LLM-as-judge harness: faithfulness 4.56/5, quality 4.62/5, refusal decisions 9/9 exact, and status-match 29/41 where every miss is a one-notch ok/partial completeness call (0 judge failures, 0 errors). Classifier axis macro-F1 0.92; refusal precision and recall 1.0. Every number traces to a committed artifact under [`eval/results/`](eval/results/).
- **Honest eval methodology.** A judge-correlation study tested a cheap local 7B judge against Sonnet and rejected it (quality Spearman 0.11). Sonnet judges everything; the decision and its gates are checked into [`eval/judge_config.yaml`](eval/judge_config.yaml).
- **Does the critic earn its cost? Measured.** A four-arm paired A/B campaign (critic on/off, a critic-analyst rebuttal loop, a tiered agentic analyst) with bootstrap CIs: the critic adds +0.07 quality at +$0.086/case with faithfulness flat; the rebuttal loop posts the only statistically significant quality gain (+0.10) but nudges faithfulness down, so it ships off by the pre-registered rule; the agentic analyst loses on both and ships off. Numbers and the decision reasoning are in [ARCHITECTURE.md](ARCHITECTURE.md#ab-does-the-critic-earn-its-cost).
- **Durable by construction, proven by SIGKILL.** A Postgres checkpointer writes state at every super-step; `/runs/{id}/resume` re-drives from the last checkpoint, and re-run nodes hit a canonical-JSON LLM cache. A kill-resume suite in CI SIGKILLs runs mid-LLM-call, mid-fan-out, and mid-critic-turn: every resume finishes with a byte-identical report and zero duplicate API calls.
- **Real cost accounting.** Every LLM call writes a trace row with token counts and two dollar figures: billed (notional) and effective (zero on a cache replay), so A/B pairing stays fair while actual spend stays visible. A judged run averages $0.124/case, and the critic is the measured cost driver (68% of arm spend).
- **Two surfaces.** FastAPI with an SSE stream of node events (watch the agent work), and an MCP server exposing the six tools plus a high-level `compare_companies` for Claude Desktop.
- **Typed and gated.** Pydantic v2 state with a discriminated citation union and parallel-write-safe reducers; mypy strict on the typed core; ruff lint and format; CI on every push.

## Demo

![Quorum demo](assets/demo.gif)

The recording is a replay of a real run, committed at [`eval/fixtures/demo_replay.jsonl`](eval/fixtures/demo_replay.jsonl) (regenerate the GIF with [`assets/demo.tape`](assets/demo.tape)). Reproduce it with no API key, no server, no Docker:

```
uv run python scripts/demo.py --replay --step 0.45
```

What it shows: classify -> resolve -> two analysts fan out in parallel -> assess -> the critic re-derives the quant claims with 8 tool calls against Postgres and flags 3 of them (two narratives whose dollar figures check out but overstate the story, one citation-coverage gap) -> synthesize acts on every flag. The COST line shows the billed-vs-effective split: the run's notional cost is $0.19, actual spend $0.0004 - every model call except the never-cached classifier replayed from the local disk cache.

To run it live (and record a new fixture with `--record`):

```
# terminal 1 - start the API (Anthropic key is injected by ./secret-run, never printed)
./secret-run uv run uvicorn quorum.api.main:app --port 8000

# terminal 2 - stream a comparison and watch the agent work
uv run python scripts/demo.py "Compare Coca-Cola and PepsiCo on profitability and growth." --step 0.5 --cost
```

## How it works

Quorum is a single FastAPI service that drives a LangGraph agent graph. There
is no queue or worker tier - a `/compare` request runs the graph inline and
streams node events over SSE.

```mermaid
flowchart LR
    C(["Client / MCP host"])
    C -->|"POST /compare (SSE stream)"| API
    C -->|"POST /runs/:id/resume"| API
    C -->|"MCP protocol"| MCP

    subgraph api_tier["API tier - FastAPI (stateless)"]
        API["FastAPI<br/>SSE - ready checks"]
        MCP["MCP server<br/>mirrors the tools"]
    end

    API -->|"invoke, stream node events"| GRAPH

    subgraph runtime["LangGraph runtime"]
        GRAPH["Agent graph<br/>classify - resolve - plan - analyze<br/>assess - critic - synthesize"]
        TOOLS["Tools<br/>search_filings - get_financial_concept<br/>get_filing_section - resolve_company"]
        GRAPH <-->|"tool calls"| TOOLS
    end
    MCP -.->|"same tools"| TOOLS

    GRAPH -->|"route by role"| ROUTER{{"Model router"}}
    ROUTER -.->|"analyst / synthesizer / critic / judge"| ANTH(["Claude Sonnet"])
    ROUTER -.->|"classifier fallback"| HAIKU(["Claude Haiku"])
    ROUTER -.->|"classifier / legwork (gpu profile)"| VLLM(["Qwen 2.5 7B AWQ<br/>local vLLM"])

    GRAPH <-->|"checkpoints - trace_events"| PG[("Postgres 16")]
    TOOLS <-->|"XBRL facts"| PG
    TOOLS <-->|"embed + hybrid search"| QD[("Qdrant<br/>BGE-M3 dense + sparse")]
    GRAPH <-->|"replay hits"| CACHE[("diskcache<br/>LLM cache")]

    classDef store fill:#eef2ff,stroke:#6366f1,color:#1e1b4b;
    classDef ext fill:#fff7ed,stroke:#fb923c,color:#7c2d12;
    class QD,PG,CACHE store;
    class ANTH,HAIKU,VLLM ext;
```

The graph itself branches, re-plans only the weak axes, and fact-checks the
draft before it writes:

```mermaid
flowchart TB
    Q(["question"]) --> CLS["classify"]
    CLS -->|"out of scope / no axis"| REF["refuse"]
    CLS --> RES["resolve"]
    RES -->|"fewer than 2 in-corpus tickers"| REF
    RES --> PLAN["plan"]
    PLAN -->|"Send(axis) x N - parallel fan-out"| AX["analyze_axis"]
    AX --> ASSESS["assess"]
    ASSESS -->|"any axis weak? re-plan weak axes only, within budget"| PLAN
    ASSESS -->|"all grounded / budget spent"| CRIT["critic<br/>agentic tool loop, 5 turns / 90s"]
    CRIT --> SYN["synthesize<br/>drops / softens / counter-cites every flagged claim"]
    SYN --> DONE(["END"])
    REF --> DONE

    classDef terminal fill:#fef2f2,stroke:#ef4444,color:#7f1d1d;
    class REF terminal;
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

41-case gold set, judged by Sonnet ([`eval/judge_config.yaml`](eval/judge_config.yaml)), default configuration:

| Metric | Value |
|--------|-------|
| Refusal decisions (answer vs refuse)            | 9 / 9 exact    |
| Completeness match, answered cases (ok / partial) | 20 / 32      |
| Overall status match                            | 29 / 41 (0.71) |
| Faithfulness mean (32 answered cases)           | 4.56 / 5       |
| Quality mean (41 cases)                         | 4.62 / 5       |
| Faithfulness / quality judge failures           | 0 / 0          |

All 12 status misses are one-notch `ok`/`partial` completeness calls (8 where the
system was more conservative than the gold label, 4 less); no wrong refusals, no
errors, no crashes. The refusal boundary and the completeness boundary are scored
separately above because they fail for different reasons - see [Limitations](#limitations).

Classifier, deterministic scoring over the full gold set:

| Metric | Value |
|--------|-------|
| Axis macro-F1                         | 0.92 |
| Axis exact-set-match                  | 0.85 |
| Refusal recall / precision / accuracy | 1.00 / 1.00 / 1.00 |

A prompt-injection red team (11 attack vectors plus a benign control) plants adversarial text into the retrieval corpus under a matching ticker and section, then drives each probe through the full graph: **0 leaks over 9 measured vectors** (2 unmeasured, control clean), with the critic engaging on nearly every case. Harness and cases are in [`scripts/run_injection_eval.py`](scripts/run_injection_eval.py) and [`eval/datasets/injection_v1.yaml`](eval/datasets/injection_v1.yaml).

Faithfulness is deterministic for quant citations (value + unit + period checked against Postgres) and LLM-judged for qual citations. The judge-correlation study rejected a cheap local 7B judge on quality (Spearman 0.11 against Sonnet) and kept Sonnet as the sole judge. The critic-on-vs-off A/B ran as a four-arm campaign: the critic adds +0.067 quality (95% CI -0.037 to +0.183) at +$0.086/case with faithfulness flat, and synthesis acted on all 56 flagged claims (incorporation rate 1.0). Per-case artifacts behind every number (reports, scores, citations, paired compares) are committed under [`eval/results/campaign-critic/`](eval/results/campaign-critic/) and [`eval/results/campaign-compares/`](eval/results/campaign-compares/). Full tables, the cost breakdown, and the honest analysis behind each number are in [ARCHITECTURE.md](ARCHITECTURE.md#eval-harness).

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
uv run python scripts/run_ab_compare.py eval/runs/campaign-baseline eval/runs/campaign-critic --cost
```

A judged run averages $0.124/case (p50 $0.125, p95 $0.282); refusals short-circuit to ~$0.001. The critic is the cost driver (~$0.027/turn, 68% of arm spend). See [ARCHITECTURE.md](ARCHITECTURE.md#cost) for the per-node breakdown, the billed-vs-effective split, and the caching story.

## Engineering

- **Types as a gate.** Pydantic v2 state with a discriminated `Citation` union and reducers that are safe under the parallel fan-out. mypy strict on the typed core (`state`, `graph`, `tools`, `models`, `cache`).
- **CI on every push.** ruff lint, ruff format check, mypy, and the unit + smoke suites ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Integration, GPU, and token-billing eval suites are deliberately out of CI.
- **Resume that cannot silently rot.** The checkpointer uses an explicit msgpack allowlist; a completeness test fails the build if a state model is added without registering it, and another proves the allowlist blocks stray types on deserialize.
- **pre-commit** mirrors the CI lint and format gates.

## Limitations

A few honest edges, with the full list in [ARCHITECTURE.md](ARCHITECTURE.md#limitations):

- The corpus is fixed (latest 10-K + four 10-Qs per company). A question whose scope exceeds that window, or asks for a breakout the XBRL facts do not isolate, is answered on the available slice; flagging that under-scope and downgrading to `partial` is a v2 item (4 of the 12 status mismatches).
- The `assess` node over-flags well-grounded qualitative axes as weak (8 of the 12 status mismatches); the grounding heuristic is tuned for quant-fact density and under-credits qual evidence. This is the main reason status match reads 29/41.
- Sonnet writes the reports and Sonnet judges them. Same-model self-preference is disclosed, not measured; what bounds it is that quant faithfulness and status match are deterministic code, not judge opinion.
- The local Qwen classifier is a portfolio statement, not a v1 cost win: Haiku is cheaper than the GPU time at this scale. The honest framing is in the architecture doc.
- The prompt-injection result (0 leaks over 9 measured vectors) is a single-run red team: 2 vectors are unmeasured, N is small, and there is no explicit data/instruction delimiting layer yet. Delimiting plus a no-injection counterfactual for the grounding vector are v2 items; the current result is that the grounded-by-construction design and the critic already resist every measurable vector.

## More

- [ARCHITECTURE.md](ARCHITECTURE.md) - node-by-node graph, state schema, retrieval, the full eval harness, cost, and limitations.
- [`scripts/demo.py`](scripts/demo.py) - the live trajectory renderer used in the demo above.
