from __future__ import annotations

import pytest

from quorum.graph.nodes.analyze_axis import (
    _extract_first_json_object,
    _parse_analyst_output,
)


def test_bare_json_object() -> None:
    assert _parse_analyst_output('{"grounding": "ok"}') == {"grounding": "ok"}


def test_markdown_fence_with_json_tag() -> None:
    raw = '```json\n{"grounding": "ok"}\n```'
    assert _parse_analyst_output(raw) == {"grounding": "ok"}


def test_markdown_fence_without_json_tag() -> None:
    raw = '```\n{"grounding": "ok"}\n```'
    assert _parse_analyst_output(raw) == {"grounding": "ok"}


def test_prose_prefix_before_object() -> None:
    # The smoke-run regression: Sonnet routinely prepends a sentence.
    raw = 'Here is the analysis:\n\n{"grounding": "ok", "comparison": "..."}'
    assert _parse_analyst_output(raw) == {"grounding": "ok", "comparison": "..."}


def test_prose_suffix_after_object() -> None:
    raw = '{"grounding": "ok"}\n\nLet me know if you need more detail.'
    assert _parse_analyst_output(raw) == {"grounding": "ok"}


def test_prose_prefix_and_suffix() -> None:
    raw = 'Analysis below.\n```json\n{"grounding": "weak"}\n```\nDone.'
    assert _parse_analyst_output(raw) == {"grounding": "weak"}


def test_nested_object_preserved() -> None:
    raw = '{"per_company": {"AAPL": {"values": {"FY2024": "391B"}}}}'
    assert _parse_analyst_output(raw) == {"per_company": {"AAPL": {"values": {"FY2024": "391B"}}}}


def test_brace_inside_string_does_not_break_match() -> None:
    # The walker treats "..." as opaque; braces inside strings must not affect
    # the depth counter.
    raw = '{"comparison": "a sample object looks like {x: y}"}'
    assert _parse_analyst_output(raw) == {"comparison": "a sample object looks like {x: y}"}


def test_escaped_quote_inside_string() -> None:
    raw = r'{"comparison": "they said \"hi\" then left"}'
    out = _parse_analyst_output(raw)
    assert out["comparison"] == 'they said "hi" then left'


def test_first_object_wins_when_multiple_present() -> None:
    # If Sonnet writes two objects (rare; usually a JSON example then the real
    # output), prefer the first balanced one. Acceptable contract: as long as
    # ONE is balanced and valid, the parser does not raise.
    raw = '{"grounding": "ok"} {"ignored": true}'
    assert _parse_analyst_output(raw) == {"grounding": "ok"}


def test_raises_when_no_object_present() -> None:
    with pytest.raises((ValueError, Exception)):
        _parse_analyst_output("hello world, no JSON here")


def test_raises_on_truncated_object() -> None:
    # Unbalanced braces -> _extract_first_json_object returns None -> ValueError.
    with pytest.raises(ValueError):
        _parse_analyst_output('{"grounding": "ok"')


def test_raises_on_empty_input() -> None:
    with pytest.raises(ValueError):
        _parse_analyst_output("")


def test_stray_close_brace_before_open_is_ignored() -> None:
    # A "}" that appears before any "{" must not crash the depth counter.
    raw = 'this is } odd but then {"grounding": "ok"}'
    assert _parse_analyst_output(raw) == {"grounding": "ok"}


def test_extract_returns_none_for_no_object() -> None:
    assert _extract_first_json_object("nothing to see") is None


def test_extract_handles_unicode_inside_string() -> None:
    raw = '{"comparison": "company é reported"}'
    out = _parse_analyst_output(raw)
    assert out["comparison"] == "company é reported"
