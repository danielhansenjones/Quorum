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
