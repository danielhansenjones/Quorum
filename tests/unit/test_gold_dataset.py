from __future__ import annotations

from pathlib import Path

from quorum.config.companies import CIK_BY_TICKER
from quorum.eval.runner import load_gold
from quorum.graph.axis_config import SUPPORTED_AXES

_GOLD = Path("eval/datasets/v1/gold.yaml")


def test_gold_parses_and_is_nontrivial() -> None:
    cases = load_gold(_GOLD)
    assert len(cases) >= 40


def test_gold_ids_unique() -> None:
    ids = [c.id for c in load_gold(_GOLD)]
    assert len(ids) == len(set(ids))


def test_gold_tickers_in_corpus() -> None:
    for c in load_gold(_GOLD):
        for t in c.expected_tickers:
            assert t in CIK_BY_TICKER, f"{c.id}: ticker {t} not in corpus"


def test_gold_axes_supported() -> None:
    for c in load_gold(_GOLD):
        for a in c.expected_axes:
            assert a in SUPPORTED_AXES, f"{c.id}: axis {a} not supported"


def test_gold_status_values_valid() -> None:
    for c in load_gold(_GOLD):
        assert c.expected_status in ("ok", "partial", "refused"), c.id


def test_gold_refusals_have_no_axes() -> None:
    for c in load_gold(_GOLD):
        if c.expected_status == "refused":
            assert not c.expected_axes, f"{c.id}: refusal should have no axes"


def test_gold_covers_each_category() -> None:
    statuses = {c.expected_status for c in load_gold(_GOLD)}
    assert {"ok", "partial", "refused"} <= statuses
