from __future__ import annotations

from quorum.tools.resolve_company import resolve_company


def test_exact_ticker() -> None:
    r = resolve_company("AAPL")
    assert r is not None
    assert r.ticker == "AAPL"
    assert r.cik == "320193"


def test_ticker_case_insensitive() -> None:
    r = resolve_company("aapl")
    assert r is not None
    assert r.ticker == "AAPL"


def test_exact_name() -> None:
    r = resolve_company("Apple Inc.")
    assert r is not None
    assert r.ticker == "AAPL"


def test_name_without_suffix() -> None:
    r = resolve_company("Apple")
    assert r is not None
    assert r.ticker == "AAPL"


def test_name_lowercase_with_inc() -> None:
    r = resolve_company("apple inc")
    assert r is not None
    assert r.ticker == "AAPL"


def test_cik_padded() -> None:
    r = resolve_company("0000320193")
    assert r is not None
    assert r.ticker == "AAPL"


def test_cik_unpadded() -> None:
    r = resolve_company("320193")
    assert r is not None
    assert r.ticker == "AAPL"


def test_out_of_corpus_returns_none() -> None:
    # Nestle is not in the v1 corpus.
    assert resolve_company("Nestle") is None
    assert resolve_company("NSRGY") is None


def test_empty_query() -> None:
    assert resolve_company("") is None
    assert resolve_company("   ") is None


def test_ambiguous_returns_none() -> None:
    # "PG" is a ticker but if we query a name that could match multiple, we
    # should return None. This validates the ambiguity guard. Our corpus
    # doesn't have natural ambiguity, so simulate via a very-short query.
    assert resolve_company("co") is None  # too generic; multiple companies end in "co"


def test_meta_resolves() -> None:
    r = resolve_company("Meta Platforms")
    assert r is not None
    assert r.ticker == "META"


def test_the_prefix_stripped_coca_cola() -> None:
    # Regression: classifier extracts "Coca-Cola"; corpus name is
    # "The Coca-Cola Company". Must resolve to KO, not refuse.
    for q in ("Coca-Cola", "coca-cola", "The Coca-Cola Company", "Coca-Cola Company"):
        r = resolve_company(q)
        assert r is not None, q
        assert r.ticker == "KO", q


def test_the_prefix_stripped_procter() -> None:
    for q in ("Procter & Gamble", "The Procter & Gamble Company", "Procter & Gamble Company"):
        r = resolve_company(q)
        assert r is not None, q
        assert r.ticker == "PG", q


def test_eli_lilly_resolves() -> None:
    # Regression: classifier extracts "Eli Lilly"; corpus name is "Eli Lilly and
    # Company". The " and company" connector left "eli lilly and" as the only
    # alias, so the natural mention refused (4 false refusals in judged-full-v1).
    for q in ("Eli Lilly", "eli lilly", "Eli Lilly and Company"):
        r = resolve_company(q)
        assert r is not None, q
        assert r.ticker == "LLY", q


def test_pharma_corpus_resolves() -> None:
    # The four pharma names must all resolve from their common mentions; these
    # cases co-occur in the gold set and any one failing forces a refusal.
    for q, ticker in (("Merck", "MRK"), ("Pfizer", "PFE"), ("Johnson & Johnson", "JNJ")):
        r = resolve_company(q)
        assert r is not None, q
        assert r.ticker == ticker, q


def test_informal_aliases_resolve() -> None:
    # Common short names the alias derivation can't reach: corpus names are
    # "Alphabet"/"Meta"/"Coca-Cola"/"PepsiCo"/"Procter & Gamble".
    for q, ticker in (
        ("Google", "GOOGL"),
        ("google", "GOOGL"),
        ("Facebook", "META"),
        ("Coke", "KO"),
        ("Pepsi", "PEP"),
        ("P&G", "PG"),
    ):
        r = resolve_company(q)
        assert r is not None, q
        assert r.ticker == ticker, q
