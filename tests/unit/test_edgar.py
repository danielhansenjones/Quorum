from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest
import respx

from quorum.ingest.edgar import EdgarFetcher, RateLimiter


def test_rate_limiter_enforces_interval() -> None:
    rl = RateLimiter(max_per_sec=10)
    start = time.monotonic()
    for _ in range(11):
        rl.acquire()
    elapsed = time.monotonic() - start
    # 11 acquisitions with 10/sec gate means at least 10 intervals = 1.0s.
    assert elapsed >= 1.0


def test_rate_limiter_does_not_oversleep() -> None:
    rl = RateLimiter(max_per_sec=10)
    start = time.monotonic()
    for _ in range(11):
        rl.acquire()
    elapsed = time.monotonic() - start
    # Generous upper bound: 11 acquisitions x 0.1s interval + scheduling jitter.
    assert elapsed < 2.5


@respx.mock
def test_fetch_submissions_caches_locally(tmp_path: Path) -> None:
    body = {"filings": {"recent": {"form": [], "accessionNumber": [], "primaryDocument": []}}}
    route = respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=body)
    )
    f = EdgarFetcher(user_agent="dev <x@example.com>", cache_dir=tmp_path)
    f.fetch_submissions("320193")
    f.fetch_submissions("320193")
    # Second call must hit cache: only one network call.
    assert route.call_count == 1


@respx.mock
def test_list_filings_caps_by_form(tmp_path: Path) -> None:
    body = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "10-Q", "10-Q", "10-Q", "10-Q", "10-K", "8-K"],
                "accessionNumber": [f"acc-{i}" for i in range(8)],
                "primaryDocument": [f"doc-{i}.htm" for i in range(8)],
                "filingDate": ["2026-01-01"] * 8,
                "reportDate": ["2025-12-31"] * 8,
            }
        }
    }
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=body)
    )
    f = EdgarFetcher(user_agent="dev <x@example.com>", cache_dir=tmp_path)
    filings = f.list_filings("320193", forms=("10-K", "10-Q"), max_per_form={"10-K": 1, "10-Q": 4})
    # The decision: 1 most-recent 10-K + 4 most-recent 10-Q. 8-K skipped.
    forms = [filing.form for filing in filings]
    assert forms.count("10-K") == 1
    assert forms.count("10-Q") == 4
    assert "8-K" not in forms


def test_user_agent_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        EdgarFetcher(user_agent="", cache_dir=tmp_path)


@respx.mock
def test_fetch_primary_doc_uses_cik_int_url(tmp_path: Path) -> None:
    # CIK in the path is unpadded integer, not zero-padded.
    expected = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-10k.htm"
    respx.get(expected).mock(return_value=httpx.Response(200, content=b"<html>filing</html>"))
    f = EdgarFetcher(user_agent="dev <x@example.com>", cache_dir=tmp_path)
    from quorum.ingest.edgar import FilingRef

    fr = FilingRef(
        cik="320193",
        accession="0000320193-25-000079",
        form="10-K",
        primary_doc="aapl-10k.htm",
        filing_date="2025-11-01",
        report_date="2025-09-27",
    )
    body = f.fetch_primary_doc(fr)
    assert body == b"<html>filing</html>"
