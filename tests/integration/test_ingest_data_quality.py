from __future__ import annotations

import socket
import statistics
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from quorum.ingest.chunk import BGE_M3_TOKENIZER_NAME, DEFAULT_TARGET_TOKENS
from quorum.ingest.edgar import EdgarFetcher
from quorum.ingest.parse import (
    ITEM_TO_CANONICAL,
    ITEM_TO_CANONICAL_10Q,
    parse_filing_html,
)
from quorum.ingest.qdrant_writer import COLLECTION_NAME

pytestmark = pytest.mark.integration

_AAPL_CIK = "320193"
_AAPL_KNOWN_10K_ACCESSION = "0000320193-25-000079"
_EDGAR_CACHE_DIR = Path("data/edgar")
_SAMPLE_CIKS_FOR_PARSE = ["320193", "789019", "21344"]  # AAPL, MSFT, KO


def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _local_fetcher() -> EdgarFetcher:
    return EdgarFetcher(
        user_agent="Quorum Integration Tests test@example.com",
        cache_dir=_EDGAR_CACHE_DIR,
    )


def test_aapl_latest_10k_accession_matches_edgar_index() -> None:
    # Phase 3a: EDGAR list_filings returns the known AAPL 10-K accession.
    # Reads from on-disk submissions cache, no network.
    if not _EDGAR_CACHE_DIR.exists():
        pytest.skip("EDGAR cache not present; run the ingest first")
    fetcher = _local_fetcher()
    try:
        filings = fetcher.list_filings(_AAPL_CIK, forms=("10-K",), max_per_form={"10-K": 1})
    finally:
        fetcher.close()
    assert len(filings) == 1
    assert filings[0].form == "10-K"
    assert filings[0].accession == _AAPL_KNOWN_10K_ACCESSION


def test_parse_5_filings_across_3_companies_finds_expected_sections() -> None:
    # Phase 3b: each company's most recent 10-K must surface the canonical Item 1,
    # 1A, and 7 sections. Any absences are logged via pytest.skip-style messages
    # but a missing section in all 3 is a hard fail.
    if not _EDGAR_CACHE_DIR.exists():
        pytest.skip("EDGAR cache not present; run the ingest first")
    fetcher = _local_fetcher()
    required = {
        "item_1_business",
        "item_1a_risk_factors",
        "item_7_mda",
    }
    seen_per_company: dict[str, set[str]] = {}
    parsed_filings = 0
    try:
        for cik in _SAMPLE_CIKS_FOR_PARSE:
            filings = fetcher.list_filings(
                cik, forms=("10-K", "10-Q"), max_per_form={"10-K": 1, "10-Q": 1}
            )
            for filing in filings:
                html = fetcher.fetch_primary_doc(filing)
                sections = parse_filing_html(html, form=filing.form)
                seen_per_company.setdefault(cik, set()).update(s.name for s in sections)
                parsed_filings += 1
    finally:
        fetcher.close()

    assert parsed_filings >= 5, f"only parsed {parsed_filings} filings"
    for required_section in required:
        coverage = [cik for cik, names in seen_per_company.items() if required_section in names]
        assert coverage, (
            f"section {required_section!r} missing in all 3 companies; "
            f"per-company sections: {seen_per_company}"
        )


def test_canonical_section_names_match_parser_constants() -> None:
    # Phase 3b cross-check: every chunk in Qdrant carries a section name that
    # appears in our canonical maps. Catches drift between parser and writer.
    if not _tcp_reachable("localhost", 6333):
        pytest.skip("qdrant not reachable")
    client = QdrantClient(url="http://localhost:6333")
    canonical = set(ITEM_TO_CANONICAL.values()) | set(ITEM_TO_CANONICAL_10Q.values())
    seen_sections: set[str] = set()
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=next_offset,
            with_payload=["section"],
            with_vectors=False,
        )
        for p in points:
            if p.payload:
                seen_sections.add(str(p.payload.get("section", "")))
        if next_offset is None:
            break
    unknown = seen_sections - canonical
    assert not unknown, f"non-canonical sections in Qdrant: {unknown}"


def test_chunk_token_counts_within_band() -> None:
    # Phase 3c: target 750 tokens, +/- 20% (so [600, 900]). The last chunk per
    # (accession, section) is allowed to be smaller (sections shorter than the
    # window naturally produce one short chunk; we don't pad). We compute the
    # 95% threshold on body chunks only - the last-chunk-per-section is
    # excluded. The upper bound (no chunk exceeds 900) holds for ALL chunks.
    if not _tcp_reachable("localhost", 6333):
        pytest.skip("qdrant not reachable")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BGE_M3_TOKENIZER_NAME)
    client = QdrantClient(url="http://localhost:6333")

    lower = int(DEFAULT_TARGET_TOKENS * 0.8)
    upper = int(DEFAULT_TARGET_TOKENS * 1.2)

    by_section: dict[tuple[str, str], list[tuple[int, int]]] = {}
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=next_offset,
            with_payload=["text", "accession", "section", "chunk_id"],
            with_vectors=False,
        )
        for p in points:
            if not p.payload:
                continue
            text = str(p.payload.get("text", ""))
            if not text:
                continue
            chunk_id = str(p.payload.get("chunk_id", ""))
            try:
                ordinal = int(chunk_id.rsplit(":", 1)[1])
            except (IndexError, ValueError):
                ordinal = 0
            n_tok = len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])
            key = (str(p.payload.get("accession", "")), str(p.payload.get("section", "")))
            by_section.setdefault(key, []).append((ordinal, n_tok))
        if next_offset is None:
            break

    assert by_section, "no chunks pulled from Qdrant"

    body_counts: list[int] = []
    tail_counts: list[int] = []
    all_counts: list[int] = []
    for _, chunks in by_section.items():
        chunks.sort()  # by ordinal asc
        last_ord = chunks[-1][0]
        for ordinal, n in chunks:
            all_counts.append(n)
            if ordinal == last_ord:
                tail_counts.append(n)
            else:
                body_counts.append(n)

    # Upper bound holds for every chunk; chunker MUST cap at target_tokens.
    assert max(all_counts) <= upper, f"max chunk size {max(all_counts)} > {upper}"
    # No chunk should exceed BGE-M3 max sequence length (8192).
    assert max(all_counts) <= 8192

    # Body chunks (all but the final per section) hit the lower bound 95% of
    # the time. Body chunks come from full target_tokens windows; they should
    # almost always be exactly target_tokens.
    if body_counts:
        in_band = sum(1 for n in body_counts if lower <= n <= upper)
        ratio = in_band / len(body_counts)
        median = statistics.median(body_counts)
        assert ratio >= 0.95, (
            f"body in_band={ratio:.3f} (n_body={len(body_counts)}, "
            f"n_tail={len(tail_counts)}, median_body={median})"
        )
