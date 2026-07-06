# Quorum - Architecture

The technical companion to the [README](README.md). Node-by-node graph, state schema, retrieval, the eval harness, cost accounting, and the honest limitations.

## The graph

```
question
  |
  v
[classify] --out_of_scope OR no axis--> [refuse] --> END
  | axes + companies_raw
  v
[resolve] --fewer than 2 in-corpus companies--> [refuse] --> END
  | tickers
  v
[plan] <----------------------------------+
  | Send(axis) x N (parallel fan-out)      | re-plan only the
  v                                        | under-grounded axes
[analyze_axis]  [analyze_axis]  ...        | (while budget remains)
  \________________|________________/      |
                   v                        |
              [assess] --any axis weak?-----+
                   | all grounded OR budget exhausted
                   v
              [critic] --timeout / failure----+
                   | critique complete        |
                   v                          |
              [synthesize] <------------------+
                   |
                   v
                  END
```

### Nodes

- **classify** - extracts axes and raw company mentions; routes out-of-scope or axis-less requests straight to `refuse`. Runs on the local Qwen classifier when `VLLM_URL` is set, Haiku otherwise.
- **resolve** - maps company mentions to in-corpus tickers; refuses if fewer than two resolve (a comparison needs two sides).
- **plan** - builds one `AxisTask` per axis. Quant axes get XBRL-fact retrieval; qual axes get semantic search. On re-plan, it rebuilds only the axes `assess` marked weak.
- **analyze_axis** - the parallel fan-out. One Sonnet analyst per axis, each writing a per-company finding and a comparison grounded in retrieved evidence, with citations. Traces itself per branch.
- **assess** - reads grounding across axes and routes: back to `plan` (weak axes remain and budget allows), to `critic` (all grounded), or straight to `synthesize` (critic disabled).
- **critic** - the agentic step. A bounded tool loop (5 turns / 90s) that re-checks claims against the same XBRL facts and filing text, emitting flagged claims with reasons. On timeout or failure it is bypassed and synthesis proceeds without it.
- **synthesize** - writes the final markdown report and acts on every flag: drops, softens, or counter-cites. Sets `ok` or `partial`.
- **refuse** - the terminal out-of-scope / under-resolved path.

### Conditional edges and the budget

`after_assess` is the only multi-way branch. Re-plan budget: `max_replans = 2`, with a LangGraph step ceiling of `remaining_steps = 2 * num_axes` as the runaway-loop safety net. Two re-plans recover one weak axis without unbounded iteration. The critic has its own 5-turn / 90s containment so the agent loop cannot stall the graph; a timeout routes to `synthesize` rather than failing the run.

The critic node is conditional. With `critic_enabled=False` (`build_graph`), the node is not added and the all-grounded route from `assess` short-circuits to `synthesize`. That toggle is the off-arm for the A/B study below.

### Default-off arms: the rebuttal loop and the agentic analyst

Two further graph features are wired and unit-tested but ship default-off; each is an arm in the A/B campaign and stays off until the paired run shows it earns its cost. Every result in this doc is the default configuration (critic on, both off).

- **Rebuttal loop** (`rebuttal_enabled=True`). When the critic flags claims and step budget remains, `route_after_critic` sends the run to a `rebut` node instead of synthesis: each flagged axis's analyst gets one pass to defend, retract, or revise every flagged claim, citing only existing evidence rows, and the critic then re-checks with the rebuttals in context (`critic -> rebut -> critic`, capped at one rebuttal round - `rebuttals` is last-write-wins state, so a second round would overwrite the first and lose retractions; `remaining_steps` stays as the backstop). A malformed or missing disposition falls through to synthesize's drop/soften handling rather than being guessed.
- **Agentic analyst** (`agentic_analyst=True`). Replaces the single-shot analyst with a tiered pair: a cheap legwork model (Haiku) runs a bounded tool loop over the same retrieval tools (`graph/agent_loop.py`, the critic's turn / wall-clock containment pattern), then Sonnet writes and cites over only the gathered evidence, so the write step matches the single-shot analyst by construction. The loop speaks the Anthropic wire format only: with `VLLM_URL` set the legwork role would route to the local Qwen, so the loop fails fast with a surfaced reason (log + trace event) rather than a swallowed protocol error; wiring the OpenAI protocol into the loop is a follow-up. Legwork failure or empty evidence falls back to the single-shot analyst - loudly, since a silent fallback would measure the agentic arm as the baseline.

Both are `build_graph` flags and `scripts/run_smoke_eval.py` arms (`--rebuttal`, `--agentic`).

## State schema

State is a Pydantic v2 model. Two design points matter:

- **Discriminated `Citation` union.** A citation is either a `QuantCitation` (ticker, accession, concept, value, unit, period - checkable against Postgres) or a `QualCitation` (filing passage - checkable against Qdrant). The discriminator keeps the two faithfulness paths (deterministic vs LLM-judged) clean.
- **Parallel-write-safe reducers.** The fan-out has N analyst branches writing back concurrently. The `axis_results` reducer merges branch returns without lost updates, which is what makes `Send`-based fan-out safe to checkpoint.

`request_id` is idempotent at entry, so re-entry does not duplicate work.

## Resume and durability

`AsyncPostgresSaver` is wired at boot. LangGraph writes a checkpoint at every super-step; a single-axis run produces roughly ten checkpoints across entry / classify / resolve / plan / analyze_axis / assess / critic / synthesize.

`GET /runs/{request_id}/resume` calls `ainvoke(None, config)` with the run's `thread_id`, which replays from the last completed node. At most the interrupted node re-runs. Because every LLM call routes through the canonical-JSON cache (key over model + messages + system prompt + tool schemas + params), a re-run node's model calls hit the cache and fire zero new API calls when inputs are unchanged. State is a Pydantic model and `request_id` is idempotent, so re-entry does not double-count.

The checkpointer serializer uses an explicit msgpack allowlist (`CHECKPOINT_MODELS`). `tests/unit/test_checkpoint_allowlist.py` enforces two things: every state model is registered (a missing one would silently break resume), and an unregistered type is blocked on deserialize (proving the allowlist does real work). This is the guard that keeps resume from rotting as state evolves.

The durability claim is proven, not asserted: a SIGKILL suite (`tests/kill_resume/`, its own CI job against a Postgres service container) kills a subprocess run at clean node boundaries, mid-LLM-call, mid-fan-out, mid-re-plan, and mid-critic-turn, then resumes. Resumed runs finish with byte-identical reports and zero duplicate API calls (scripted model fakes plus the disk cache make both assertable). The empirical answers worth recording:

- LangGraph checkpoints once per superstep, so all fan-out branch writes land in the single join checkpoint. Partial fan-out recovery comes from task-level pending writes plus cache-hit re-runs, not per-branch checkpoints.
- Node trace rows are written at node completion: a killed attempt leaves no row, and resume does not double-count. On a disk-cache replay the trace row keeps the notional billed cost and records zero effective spend, so cost pairing survives a resume.
- The suite runs the checkpointer with `durability="sync"`. The production default is async, where a hard kill can lose the newest superstep's checkpoint; resume then re-runs that superstep and the cache absorbs the cost.
- A state-schema widening smoke (resume an old checkpoint under a schema with a new optional field) loads cleanly: langgraph leaves the new channel absent in old checkpoint values and pydantic fills the default at construction time.

## Retrieval and data layer

- **Qdrant, hybrid.** BGE-M3 dense vectors plus a learned sparse vector, fused with RRF at query time. BGE-M3 embeds on CPU. ColBERT multi-vector output is neither computed nor stored (the embedder passes `return_colbert_vecs=False`; the collection has no multivector field); the measured case for keeping it that way is in [Retrieval](#retrieval) below.
- **Postgres, facts and traces.** XBRL company facts, the LangGraph checkpointer, and the `trace_events` table that drives both the eval harness and cost accounting.
- **Concept normalization.** Cross-company comparison needs the same metric across different XBRL tags - PG's `us-gaap:Revenues` vs AAPL's `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax`. `config/concept_aliases.yaml` holds curated fallback chains per concept, with per-ticker overrides. It is hand-curated for the 12-company corpus.
- **Ingest** is offline and separate from the graph: EDGAR fetcher -> HTML parser with Item segmentation (BeautifulSoup + lxml) -> chunker -> Qdrant writer and Postgres facts.

## Eval harness

The harness runs the gold set through the graph, writes one JSON per case plus a `summary.json`, and optionally scores each case. Output lands in `eval/runs/<run_id>/` (local, gitignored); the canonical artifacts behind the numbers cited here are committed under `eval/results/`.

### Faithfulness and quality

Faithfulness is deterministic for quant citations (value + unit + period checked against Postgres, which floors grounded quant axes near 5.0) and LLM-judged for qual citations. Quality is one Sonnet rubric pass over the whole report.

Full 41-case judged run ([`eval/results/campaign-critic/summary.json`](eval/results/campaign-critic/summary.json), Sonnet judge, the default configuration - critic on, rebuttal and agentic off):

| Metric | Value |
|--------|-------|
| Status match (vs expected ok / partial / refused) | 29 / 41 (0.71) |
| Faithfulness mean (32 answered cases) | 4.56 / 5 |
| Quality mean (41 cases)               | 4.62 / 5 |
| Faithfulness / quality judge failures | 0 / 0 |

The faithfulness mean is pulled down by the interpretive `risk_factors` axis (per-case 3.2-3.5) and two weakly-cited cases (`jnj_pfe_profitability` 2.3, `gross_margin_googl` 2.3); quant-grounded axes sit near 5.0. The twelve status mismatches split into the two known issues, not regressions: eight qualitative axes the `assess` node over-flags as weak (reported `partial`, gold `ok`), and four questions whose temporal or segment scope exceeds the corpus answered on the available slice without flagging the shortfall (reported `ok`, gold `partial`). Both are in [Limitations](#limitations). Faithfulness covers the 32 answered cases; refusals carry no report. The metric excludes-and-counts judge errors (`faithfulness_judge_failures`, 0 here), so means from before that change are not bit-comparable.

### Classification and refusal

Deterministic scoring over the full gold set (`scripts/run_classification_eval.py`, baseline in `eval/baselines/classification_v1.json`):

| Metric | Value |
|--------|-------|
| Axis macro-F1 | 0.92 |
| Axis exact-set-match | 0.85 |
| F1 - risk_factors / leverage / profitability / growth | 1.00 / 0.89 / 0.91 / 0.87 |
| Refusal recall / precision / accuracy | 1.00 / 1.00 / 1.00 |

The classifier and resolver separate refuse-vs-answer perfectly on the gold set (9 true refusals caught, 0 false refusals). The remaining axis error is `growth` over-prediction (recall 1.0, precision 0.77). An earlier 0.69 refusal precision was a resolver alias bug (`"Eli Lilly"` not matching `"Eli Lilly and Company"`), since fixed.

### Retrieval

Measures the Qdrant index directly: hybrid (dense + sparse RRF, the production configuration) against dense-only and sparse-only, on a labeled set built for the two query populations the system actually issues ([`eval/datasets/retrieval_v1.yaml`](eval/datasets/retrieval_v1.yaml), 55 queries, 372 labeled positives). The 12 `planner` queries are the fixed `risk_factors` axis query per ticker; the 43 `freeform` queries are critic/MCP-style topic probes authored against chunk text and paraphrased to avoid verbatim lexical matches. Candidates were pooled from the union of all three arms' top-10 (772 judged pairs); labeling criteria and protocol are in the dataset header. Runner: `scripts/run_retrieval_eval.py` (deterministic, no LLM calls; artifact in [`eval/results/retrieval-v1/`](eval/results/retrieval-v1/)).

Freeform (43 queries; success@k = any positive in top k, the operative metric since positives repeat across 10-K/10-Q filings):

| Metric | hybrid | dense | sparse |
|--------|--------|-------|--------|
| success@1  | 0.81 | 0.91 | 0.79 |
| success@5  | 0.98 | 0.98 | 0.91 |
| success@10 | 1.00 | 1.00 | 1.00 |
| recall@5   | 0.67 | 0.66 | 0.65 |
| MRR        | 0.89 | 0.95 | 0.85 |

Planner (12 queries, judged precision@5 of substantive risk content): 1.00 for all three arms - identical because the section filter, not ranking, dominates. This was 0.92 until the segmentation fix: PFE scored 0.00 because last-occurrence-wins anchored its 10-K `item_1a` to cross-reference stubs while the real risk text was buried under mislabeled items (mine-safety, market-risk). The eval turned that into a number, which root-caused to a parser defect rather than a retrieval failure, got fixed (TOC-strip plus ordered boundary selection, replacing last-occurrence-wins), and re-measured to 1.00.

Decisions:

- **Hybrid stays**, narrowly. It ties dense on success@5 and leads recall@5; the analyst consumes the whole top-5, so the first-hit metrics where dense wins (success@1 0.91 vs 0.81 - a handful of queries where RRF promotes a sparse-favored near-miss to rank 1) do not change evidence sets. Dense-only is a defensible simplification; the data does not show hybrid dominance.
- **ColBERT stays out.** Every arm reaches success@10 = 1.00 on the labeled set, so a reranking stage has no headroom to buy. Adding it would cost multivector storage, per-upsert compute, collection recreation, and a re-ingest. Earlier revisions of this document called ColBERT "indexed-but-unused"; that was wrong - it was never computed or stored.

Caveats: single small corpus (2,895 indexed chunks), pooled labeling only judges what some arm surfaced (author-known positives no arm retrieved are kept as recall misses), and queries plus labels were authored with LLM assistance against chunk text with hand adjudication of disagreements.

### Judge correlation - the two-tier judge that did not survive contact

The design proposed a cheap local 7B judge for fast iteration with Sonnet as the canonical judge. `scripts/run_judge_correlation.py` re-scores the 32 answered cases from the committed [`eval/results/campaign-critic/`](eval/results/campaign-critic/) artifacts with the local Qwen-7B judge (quality rubric plus qual-citation checks, with Sonnet re-judging the qual citations as the reference) and correlates the two. Verdict: **local judge rejected, Sonnet judges everything.** The decision and gates are in `eval/judge_config.yaml`; the raw per-case pairs are in [`eval/results/judge_correlation/study.json`](eval/results/judge_correlation/study.json).

| Dimension | Spearman (local vs Sonnet) | Gate | Pass |
|-----------|----------------------------|------|------|
| Quality | 0.597 | > 0.6 | no |
| Faithfulness, qual-only | 0.46 | > 0.7 | no |
| Faithfulness, blended | 0.99 (inflated) | - | n/a |

The 7B is a near-constant scorer: 27 of 32 reports land on quality 4.0 or 4.25 while Sonnet uses 3.5-5.0, so the 0.597 hovers at the gate without carrying real signal. The blended 0.99 is inflated by construction - 23 of 32 cases sit at 5.0/5.0 because quant faithfulness is identical deterministic code on both judges; on the nine cases where qual judging actually decides the score, correlation falls to 0.46, and the local judge runs lenient on the unfaithful end (the four reports Sonnet scores 2.1-2.4, the 7B scores 3.0-3.9). Zero parse failures on the 7B, so this is calibration, not malformed output. Caveat: nine qual-only pairs is a small sample - an earlier same-day run against a stale pre-re-ingest run dir measured 0.17, so treat the correlation as indicative, not precise. Either way the gates fail and the cost-saving premise does not hold for a base 7B; Sonnet handles all judging.

### A/B: does the critic earn its cost?

The critic is the dominant cost. Whether it improves the output is a measurable question, so the harness measures it rather than assuming it:

- `build_graph(critic_enabled=...)` produces the two arms (baseline vs +critic) from the same frozen graph.
- `score_incorporation` (in `judges.py`) is a deterministic proxy for whether synthesis acted on a flag: a flagged claim is "incorporated" if it is not present verbatim in the final report (dropped, softened, or counter-cited all change the text). It distinguishes "no flags" (n=0) from "no critic" (None).
- `eval/ab_compare.py` pairs two run dirs by `case_id` and reports per-metric mean deltas with a paired bootstrap 95% CI (fixed seed, deterministic), folding in per-request cost when the trace DB is available. Run it with `scripts/run_ab_compare.py`.

The four-arm campaign ran on the full 41-case gold set, one arm per toggle combination, same commit, same judge, shared LLM cache. Per-arm artifacts are committed under `eval/results/campaign-*/`; the five paired compares under [`eval/results/campaign-compares/`](eval/results/campaign-compares/).

| Arm | Faithfulness | Quality | Status match |
|-----|--------------|---------|--------------|
| baseline (no critic)  | 4.56 | 4.53 | 31 / 41 |
| +critic (the default) | 4.56 | 4.62 | 29 / 41 |
| +rebuttal             | 4.55 | 4.66 | 29 / 41 |
| +agentic analyst      | 4.50 | 4.57 | 29 / 41 |

Paired deltas vs baseline (mean, bootstrap 95% CI):

| Arm | Faithfulness | Quality | Cost / case |
|-----|--------------|---------|-------------|
| +critic   | +0.004 [-0.006, +0.019] | +0.067 [-0.037, +0.183] | +$0.086 [+0.065, +0.108] |
| +rebuttal | -0.007 [-0.016, -0.001] | +0.104 [+0.006, +0.213] | +$0.125 [+0.096, +0.152] |
| +agentic  | -0.055 [-0.127, +0.015] | +0.012 [-0.098, +0.128] | +$0.138 [+0.103, +0.171] |

Decisions from the data, stated with the numbers rather than hidden:

- **Critic stays on.** Quality +0.067 with a CI that includes zero at n=41 - the judge-score case alone does not clear the bar. What the critic buys is the verification artifact: 56 flagged claims across 32 critiques, incorporation rate 1.0 (synthesis acted on every flag), 257/257 valid tool calls, zero timeouts. Faithfulness was already near its ceiling from the deterministic quant checks, leaving the judge little room to move. The honest summary: the critic is the product's verification story at +$0.086/case, not a measured judge-score win.
- **Rebuttal loop stays off.** The pre-registered ship rule was faithfulness flat-or-up at acceptable cost. Measured: faithfulness -0.007 with a CI excluding zero - statistically down, if microscopically. It also produced the campaign's only significant quality gain (+0.104, CI excludes zero) and the disposition data is healthy (48 flagged claims across 22 cases: 43 revised, 4 retracted, 1 defended; post-rebuttal flags dropped 56 to 28). The revise-heavy behavior explains the tension: revised claims re-word the report, and re-worded prose judges slightly less faithful. It is the most promising follow-up, but the rule says off.
- **Agentic analyst stays off.** Faithfulness -0.055 and quality below the critic arm, at the campaign's highest cost (+$0.138/case over baseline). The tiered loop itself ran clean (Haiku legwork, zero fallbacks, `run_config.models` confirms the routing), so the loss is evidence quality, not infrastructure - the cheap model gathers worse evidence than the code-driven single-shot path.

One measurement honesty note: full-campaign cache replays are near-identical but not bit-identical (retrieval-order nondeterminism causes a handful of cache misses), which moves judge means by roughly +/-0.03 between replays. The committed artifacts are one self-consistent replay set: every arm on the same commit, same cache, same judge.

### Agent-level

`eval/tool_use.py` validates the critic's recorded tool calls (argument shape, known tool names) and the runner aggregates per-run critic headlines: incorporation rate, `turns_used` distribution, timeout rate, and tool-use validity fraction.

## Cost

Every LLM call emits a `trace_events` row with real token counts and two dollar figures: `cost_dollars_billed` (the notional price of the call) and `cost_dollars_effective` (actual spend - zero when the disk cache answered). Replays and resumes keep their notional cost so A/B pairing stays comparable across warm and cold arms, while the effective column reports what was actually paid. `scripts/run_cost_report.py` aggregates per request and per node. `attempt_number` in `trace_events` is always 1 by design; attempt ordering is derived at read time (`ROW_NUMBER` over `id` per request_id + node_name), not written.

Campaign numbers (critic arm, 41 requests, `run_cost_report.py` scoped to the arm's request_ids):

| Node | $/call | notes |
|------|--------|-------|
| llm:critic      | ~$0.027/turn | dominant cost - 128 turns, 68% of arm spend |
| llm:synthesizer | ~$0.021 | one Sonnet call per report |
| llm:analyst     | ~$0.018 | one Sonnet call per axis |
| llm:classifier  | ~$0.0004 | Haiku |

Per-request: mean $0.124 across the gold set (nine refusals short-circuit near $0), p50 $0.125, p95 $0.282 (multi-axis with a five-turn critic). The critic being the cost driver is the direct input to the A/B measurement above.

Two caches matter and they are separate:

- **Local disk cache** - canonical-JSON key over model + messages + system prompt + tool schemas + params, so a prompt or tool edit misses on its own instead of replaying stale responses. Reported 100% hit rate on a re-run of an unchanged eval set (20/20 calls on a two-pass measurement, `scripts/run_cache_hitrate.py`). This is what makes resume free when inputs are unchanged. The eval judges ride the same cache, so re-judging identical reports (a crashed or repeated arm) re-bills nothing. One measured caveat: a full-campaign replay is near-free but not fully free - retrieval-order nondeterminism occasionally reorders evidence in a prompt, which misses the cache and cascades downstream; the billed-vs-effective split in the trace rows is how that residual spend is visible.
- **Anthropic prompt cache** - separate, and reads ~0 at v1 prompt sizes (system prompts are below the cache minimum).

## Local model serving

Qwen 2.5 7B Instruct (AWQ-4bit) served by vLLM, used as the classifier. AWQ-4bit fits the 16GB VRAM budget with room for the KV cache and is vLLM-native (continuous batching preserved). It is optional: with `VLLM_URL` unset, Haiku is the classifier and no GPU is required.

Honest framing: the local classifier is a portfolio statement, not a v1 cost win. Haiku costs roughly $0.0005 per classification, below the GPU-time cost of a local serve at this scale. The value is the self-hosted-inference capability, not the dollars.

## Limitations

- **Fixed corpus.** Latest 10-K plus four 10-Qs per company (~4 fiscal years). A question whose scope exceeds that window ("over the last 15 years") or asks for a breakout the XBRL facts do not isolate (advertising-segment revenue) is answered confidently on the available slice **without flagging the shortfall** - it reports `ok`, not `partial`. The campaign pins this down: `partial_long_window_tech`, `partial_insufficient_growth_for_costco`, `partial_segment_revenue`, and `partial_capex_comparison` are the four mismatches from this in the critic arm. Detecting "the ask exceeds what I grounded" and downgrading to `partial` is a v2 item. When an axis renders an explicit `*Insufficient data*` section, status does drop to `partial`; the gap is the silent under-scope case.
- **`assess` over-flags qual axes.** It marks some well-grounded qualitative axes as weak, downgrading an otherwise complete report to `partial` (8 of 41 gold cases in the critic arm, concentrated in `risk_factors` and multi-axis cases such as `pharma_risks`, `staples_risks`, `multi_axis_tech`). The grounding heuristic is tuned for quant-fact density and under-credits qual evidence; retuning it is a known follow-up and the main reason status match reads 29/41.
- **The judge shares a model with the system.** Sonnet writes the reports and Sonnet scores them, so same-model self-preference is a real risk. It is disclosed, not measured: there is no human-agreement number, by decision. What bounds it: quant faithfulness is deterministic code (value + unit + period against Postgres, no judge involved), status match is deterministic, and the one cheap alternative judge tested (local 7B) was rejected on measured grounds above. A cross-model judge re-correlation is the cheap next step if this ever needs tightening.
- **Filing text is untrusted input.** A prompt injection embedded in a 10-K would reach the analyst and critic prompts. The blast radius is bounded to prose: citations are code-built (a model cannot mint one) and the synthesizer's uncited-number strip removes unbacked figures. The red-team harness (`scripts/run_injection_eval.py`, 11 vectors + a benign control) plants adversarial text into the retrieval corpus under a matching ticker and section and drives each probe through the full graph. Measured result: **0 leaks over 9 measured vectors**, control clean, critic engaging on nearly every case; 2 vectors are unmeasured (`inj_cross_company` needs per-ticker figure attribution, `inj_grounding_flag` needs a no-injection counterfactual since a genuinely-evidenced axis also grounds `ok`). This is a single small-N run with no explicit data/instruction delimiting layer yet - delimiting plus the two counterfactuals are the v2 items. The current read is that grounded-by-construction citations plus the critic resist every measurable vector.
- **Hand-curated concept aliases.** `config/concept_aliases.yaml` is curated for the 12-company corpus. Adding tickers from new sectors means extending it and re-populating the Postgres table. At ~50+ companies, an automated XBRL-taxonomy resolver becomes worth building.
- **ColBERT reranking is not built and measured unnecessary** - all arms hit success@10 = 1.00 on the labeled retrieval set (see [Retrieval](#retrieval)). Revisit only if the corpus grows past the point where the labeled numbers stop holding.
- **Local classifier economics** - see [Local model serving](#local-model-serving). A capability demonstration, not a cost win at v1 scale.

## Repo layout

```
src/quorum/
  config/   companies, settings
  cache/    canonical-JSON LLM cache, embed cache
  trace/    structured logger, trace_events writer, cost rates
  models/   BGE-M3 embedder, router, cached chat
  ingest/   EDGAR fetcher, parser, chunker, Qdrant writer, facts, aliases
  tools/    resolve_company, concept_resolver, search, filing_section, inventory
  state/    QuorumState, AxisTask, AxisResult, Critique, Citation, reducers
  graph/    node implementations + build_graph
  mcp/      FastMCP server wrapping the tools + the compiled graph
  api/      FastAPI surface (SSE stream of node events)
  eval/     runner, judges, classification, judge_correlation, tool_use, ab_compare
config/              concept_aliases.yaml (curated XBRL fallback chains)
postgres-init/       init SQL (facts, concept_aliases, trace_events)
eval/                datasets (gold set), baselines, judge_config.yaml, results/ (committed run artifacts)
scripts/             demo.py + eval / cost / correlation / A-B runners
docker-compose.yml
tests/{unit,integration,smoke}
```
