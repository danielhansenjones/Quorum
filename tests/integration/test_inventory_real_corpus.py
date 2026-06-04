from __future__ import annotations

import socket

import pytest
from qdrant_client import QdrantClient

from quorum.config.companies import COMPANIES
from quorum.tools.inventory import list_corpus, list_filings

pytestmark = pytest.mark.integration


def _tcp_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture
def qdrant(qdrant_url: str):
    if not _tcp_reachable("localhost", 6333):
        pytest.skip("qdrant not reachable")
    return QdrantClient(url=qdrant_url)


def test_list_corpus_covers_all_v1_companies(qdrant: QdrantClient, postgres_url: str) -> None:
    # Phase 4e: every v1 ticker has non-zero facts AND non-zero chunks.
    if not _tcp_reachable("localhost", 5432):
        pytest.skip("postgres not reachable")
    entries = list_corpus(qdrant, postgres_url)
    by_ticker = {e.ticker: e for e in entries}
    expected_tickers = {c.ticker for c in COMPANIES}
    assert expected_tickers.issubset(set(by_ticker.keys())), (
        f"missing tickers: {expected_tickers - set(by_ticker.keys())}"
    )
    for ticker in expected_tickers:
        entry = by_ticker[ticker]
        assert entry.facts_count > 0, f"{ticker} has zero facts"
        assert entry.chunks_count > 0, f"{ticker} has zero chunks"


def test_list_filings_per_ticker_has_10k_and_10qs(qdrant: QdrantClient) -> None:
    # Per decisions.md #3: latest 10-K + up to 4 most recent 10-Qs per company.
    for company in COMPANIES:
        filings = list_filings(qdrant, ticker=company.ticker)
        forms = sorted({f.form for f in filings})
        assert forms == ["10-K", "10-Q"], f"{company.ticker} forms={forms}"
        assert sum(1 for f in filings if f.form == "10-K") >= 1, f"{company.ticker} missing 10-K"
        assert sum(1 for f in filings if f.form == "10-Q") >= 1, f"{company.ticker} missing 10-Q"


def test_list_filings_all_returns_full_corpus(qdrant: QdrantClient) -> None:
    filings = list_filings(qdrant)
    tickers_seen = {f.ticker for f in filings}
    expected = {c.ticker for c in COMPANIES}
    assert expected.issubset(tickers_seen), (
        f"missing tickers in list_filings(): {expected - tickers_seen}"
    )
