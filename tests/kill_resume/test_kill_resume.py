from __future__ import annotations

import pytest

from tests.kill_resume.conftest import HarnessFactory

pytestmark = pytest.mark.kill_resume

ALL_AXES = ["growth", "profitability", "risk_factors"]

# Observed 2026-07-05 (langgraph 1.2.2): all three Send branch writes land in
# the single join checkpoint (one checkpoint per superstep), not one per
# branch. Cost implication: the checkpoint alone cannot resume a partially
# complete fan-out; that recovery comes from task-level pending writes stored
# against the in-flight checkpoint, plus the LLM disk cache making any re-run
# branch near-free.
FANOUT_CHECKPOINTS = 1


def test_t0_checkpoint_granularity(kr: HarnessFactory) -> None:
    h = kr.new()
    out = h.run(dump=True)
    assert out["status"] == "ok"
    assert out["axes"] == ALL_AXES
    assert all(g == "ok" for g in out["groundings"].values())

    writes_per_checkpoint = [c["writes"] for c in out["checkpoints"]]
    for node in ("classify", "resolve", "assess", "critic", "synthesize"):
        hits = sum(node in w for w in writes_per_checkpoint)
        assert hits == 1, f"{node} written in {hits} checkpoints: {writes_per_checkpoint}"

    fanout = sum("analyze_axis" in w for w in writes_per_checkpoint)
    assert fanout == FANOUT_CHECKPOINTS, writes_per_checkpoint


def test_t1_kill_at_clean_boundary(kr: HarnessFactory) -> None:
    h = kr.new()
    p = h.start_until_hook("before:plan")
    h.sigkill(p)
    pre = h.calls()
    assert pre.count("classify") == 1

    out = h.resume()
    assert out["status"] == "ok"
    assert out["axes"] == ALL_AXES
    # The classifier is never disk-cached, so a re-run of classify would have
    # fired a fresh fake call and appended a second line.
    post = h.calls()
    assert post.count("classify") == 1

    counts = h.trace_counts()
    assert counts["classify"] == 1
    assert counts["resolve"] == 1
    assert counts["synthesize"] == 1

    # Deterministic fakes + the shared disk cache make byte-equality assertable
    # against a fresh non-killed run.
    h2 = kr.new()
    out2 = h2.run()
    assert out["report"] == out2["report"]


def test_t2_kill_mid_llm_call_resume_hits_cache(kr: HarnessFactory) -> None:
    warm = kr.new()
    warm_out = warm.run()

    h = kr.new()
    p = h.start_until_hook("in_cached_chat:analyst:profitability")
    h.sigkill(p)
    out = h.resume()
    assert out["status"] == "ok"
    # Sorted-list equality also proves the reducer upserted: exactly one
    # profitability entry, no duplicates.
    assert out["axes"] == ALL_AXES
    # The classifier ran pre-kill and is uncached; every Sonnet call on both
    # the killed and resumed legs replayed from the warm disk cache, so the
    # fake never fired.
    assert h.calls() == ["classify"]
    assert out["report"] == warm_out["report"]


def test_t5_kill_mid_synthesize(kr: HarnessFactory) -> None:
    warm = kr.new()
    warm_out = warm.run()

    h = kr.new()
    p = h.start_until_hook("in_cached_chat:synthesize")
    h.sigkill(p)
    out = h.resume()
    assert out["report"] == warm_out["report"]
    assert h.calls() == ["classify"]
    # Node rows are written after node completion, so the killed synthesize
    # attempt left no row; only the resumed attempt is traced.
    assert h.trace_counts()["synthesize"] == 1


def test_t3_kill_during_fanout_before_any_branch_completes(kr: HarnessFactory) -> None:
    # First harness of the test, so the shared cache dir is cold: analyst calls
    # must actually fire rather than replay.
    h = kr.new()
    p = h.start_until_hook("before:analyze_axis")
    h.sigkill(p)
    out = h.resume()
    assert out["status"] == "ok"
    assert out["axes"] == ALL_AXES

    analyst_axes = [c.split(":", 1)[1] for c in h.calls() if c.startswith("analyst:")]
    assert len(analyst_axes) >= 3
    assert set(analyst_axes) == set(ALL_AXES)


def test_t6_trace_rows_unique_per_node(kr: HarnessFactory) -> None:
    h = kr.new()
    p = h.start_until_hook("before:plan")
    h.sigkill(p)
    out = h.resume()
    assert out["status"] == "ok"

    # Resume must not duplicate node rows for work that completed pre-kill.
    # attempt_number is always 1 in current code, so uniqueness is asserted on
    # (request_id, node_name), not the plan doc's attempt triple.
    counts = h.trace_counts()
    for node in ("entry", "classify", "resolve", "plan", "assess", "critic", "synthesize"):
        assert counts.get(node, 0) == 1, f"{node}: {counts}"


def test_t7_cost_split_under_resume(kr: HarnessFactory) -> None:
    warm = kr.new()
    warm.run()
    warm_rows = warm.trace_cost_rows()
    warm_analyst = [r for r in warm_rows if r["node_name"] == "llm:analyst"]
    assert len(warm_analyst) == 3
    for row in warm_analyst:
        # Cold calls: real spend equals the notional price of the fake usage.
        assert row["cost_dollars_billed"] > 0
        assert row["cost_dollars_effective"] == row["cost_dollars_billed"]

    h = kr.new()
    p = h.start_until_hook("in_cached_chat:analyst:profitability")
    h.sigkill(p)
    h.resume()
    rows = h.trace_cost_rows()
    analyst = [r for r in rows if r["node_name"] == "llm:analyst"]
    # The killed profitability attempt paused before its trace row landed; the
    # other two branches completed pre-kill and the resumed attempt adds the
    # third, so no attempt is double-counted.
    assert len(analyst) == 3, rows
    for row in analyst:
        # Replays keep the notional price but spend nothing.
        assert row["cost_dollars_billed"] > 0
        assert row["cost_dollars_effective"] == 0.0

    warm_llm = [r for r in warm_rows if r["node_name"].startswith("llm:")]
    h_llm = [r for r in rows if r["node_name"].startswith("llm:")]
    assert sum(r["cost_dollars_billed"] for r in h_llm) == sum(
        r["cost_dollars_billed"] for r in warm_llm
    )
    assert sum(r["cost_dollars_effective"] for r in h_llm) == 0.0


def test_t4_kill_during_replan(kr: HarnessFactory) -> None:
    weak_env = {"KR_WEAK_FIRST_AXIS": "growth"}
    h = kr.new()
    # The shared before:plan counter counts initial_plan and revise_plan
    # arrivals together, so #2 is entry into the revise pass.
    p = h.start_until_hook("before:plan#2", extra_env=weak_env)
    h.sigkill(p)
    out = h.resume(extra_env=weak_env)

    # replan_count > 1 would mean resume re-ran the replan loop on top of a
    # checkpoint that already contained it: a checkpointer or state-schema bug.
    assert out["replan_count"] == 1, out

    comparator = kr.new()
    expected = comparator.run(extra_env=weak_env)
    assert expected["replan_count"] == 1
    # revise_plan flips growth to semantic, so the revised analyst prompt is a
    # distinct cache entry and the weak-then-ok script resolves growth to ok.
    assert expected["groundings"]["growth"] == "ok"
    assert out["groundings"]["growth"] == expected["groundings"]["growth"]
    assert out["status"] == expected["status"]


def test_t8_schema_migration_smoke(kr: HarnessFactory) -> None:
    h = kr.new()
    first = h.run()
    assert first["status"] == "ok"

    # Resume the completed thread under a schema that gained an optional field.
    # The failure this test exists to catch is a raise (pydantic
    # ValidationError) while loading the old checkpoint.
    out = h.resume(extra_env={"KR_EXTRA_STATE_FIELD": "1"})
    assert out["status"] == "ok"
    assert out["report"] == first["report"]
    # Observed 2026-07-05 (langgraph 1.2.2): the old checkpoint's values dict
    # simply lacks the new channel - langgraph does not backfill the schema
    # default into checkpoint values. Nodes still see "default" because pydantic
    # fills it when the state object is constructed.
    assert out["kr_migration_probe"] == "absent", out


def test_t9_pool_under_resume_burst(kr: HarnessFactory) -> None:
    # All five must be mid-flight before the first kill so the resumes hit the
    # pool concurrently, not serially.
    started = []
    for _ in range(5):
        h = kr.new()
        started.append((h, h.start_until_hook("before:analyze_axis")))
    for h, p in started:
        h.sigkill(p)
    resumes = [(h, h.resume_async()) for h, _ in started]
    for h, p in resumes:
        # A pool failure (psycopg OperationalError) surfaces as a nonzero exit;
        # wait_ok raises it into the assertion with stderr attached.
        out = h.wait_ok(p)
        assert out["status"] == "ok"
        assert out["axes"] == ALL_AXES


def test_t11_kill_mid_critic_multiturn(kr: HarnessFactory) -> None:
    tools_env = {"KR_CRITIC_TOOLS": "1"}
    warm = kr.new()
    warm_out = warm.run(extra_env=tools_env)
    assert warm_out["status"] == "ok"
    assert warm.calls().count("critic") == 3

    h = kr.new()
    p = h.start_until_hook("in_cached_chat:critic#3", extra_env=tools_env)
    h.sigkill(p)
    out = h.resume(extra_env=tools_env)
    assert out["report"] == warm_out["report"]
    # Every sonnet call on the killed and resumed legs replayed from the disk
    # cache - including critic turns 2-3, whose messages embed tool_result
    # payloads. That is the canonical-JSON cache-key stability claim.
    assert h.calls() == ["classify"]

    counts = h.trace_counts()
    # llm rows are written per chat call inside the node; node rows only at
    # completion. The killed leg finished critic turns 1-2 (two llm rows land
    # and survive the SIGKILL) before dying at turn 3's cache lookup; the
    # resumed leg restarts the node and re-runs all three turns. 2 + 3 = 5.
    assert counts["llm:critic"] == 5, counts
    assert counts["critic"] == 1
