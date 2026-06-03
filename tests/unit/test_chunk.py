from __future__ import annotations

import pytest

from quorum.ingest.chunk import Chunk, Chunker, chunk_filing
from quorum.ingest.parse import Section


@pytest.fixture(scope="module")
def chunker() -> Chunker:
    try:
        return Chunker(target_tokens=64, overlap_tokens=8)
    except Exception as e:  # pragma: no cover - skip path
        pytest.skip(f"BGE-M3 tokenizer unavailable: {e}")


def test_chunk_id_format() -> None:
    c = Chunk(text="x", section="item_7_mda", ordinal=3, char_start=0, char_end=1)
    assert c.chunk_id("0000320193-25-000079") == "0000320193-25-000079:item_7_mda:0003"


def test_chunker_produces_overlap(chunker: Chunker) -> None:
    text = "Apple Inc reported strong revenue growth across product lines. " * 20
    chunks = chunker.chunk_section(text, section="item_7_mda", section_char_start=100)
    assert len(chunks) >= 2
    # Adjacent chunks overlap in char space because overlap_tokens > 0.
    assert chunks[1].char_start < chunks[0].char_end


def test_chunker_advances_each_window(chunker: Chunker) -> None:
    text = "Microsoft's Intelligent Cloud segment grew. " * 50
    chunks = chunker.chunk_section(text, section="item_1_business")
    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts), "windows must not repeat a char_start"


def test_chunker_handles_short_text(chunker: Chunker) -> None:
    chunks = chunker.chunk_section("Short.", section="item_1_business")
    assert len(chunks) == 1
    assert chunks[0].text == "Short."


def test_chunker_empty_text(chunker: Chunker) -> None:
    assert chunker.chunk_section("", section="item_1_business") == []


def test_chunk_filing_excludes_item_8(chunker: Chunker) -> None:
    sections = [
        Section(name="item_7_mda", text="MD&A prose. " * 40, char_start=0, char_end=12 * 40),
        Section(
            name="item_8_financial_statements",
            text="Net income $93B " * 40,
            char_start=600,
            char_end=600 + 16 * 40,
        ),
    ]
    chunks = chunk_filing(sections, chunker=chunker)
    assert all(c.section != "item_8_financial_statements" for c in chunks)
    assert any(c.section == "item_7_mda" for c in chunks)


def test_overlap_must_be_smaller_than_target() -> None:
    with pytest.raises(ValueError):
        Chunker(target_tokens=100, overlap_tokens=100)
