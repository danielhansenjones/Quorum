from __future__ import annotations

from quorum.ingest.parse import (
    ITEM_TO_CANONICAL,
    SECTIONS_EXCLUDED_FROM_VECTOR_INDEX,
    Section,
    find_item_offsets,
    html_to_text,
    parse_filing_html,
    segment_into_sections,
)

_FAKE_10K = """<html><body>
<h1>Annual Report</h1>
<p>Item 1. Business</p>
<p>We make iPhones and services.</p>
<p>Item 1A. Risk Factors</p>
<p>Our supply chain is exposed to geopolitical risk.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Operating margin improved 200 bps year over year.</p>
<p>Item 8. Financial Statements</p>
<p>Net income $93.7B. Revenue $383.3B.</p>
<p>Item 9. Changes in Accountants</p>
<p>None.</p>
</body></html>"""

# Minimal but realistic 10-Q skeleton: TOC then real sections, both PARTs.
_FAKE_10Q = """<html><body>
<h1>Quarterly Report</h1>
<p>Table of Contents</p>
<p>PART I FINANCIAL INFORMATION</p>
<p>Item 1. Financial Statements</p>
<p>Item 2. Management's Discussion and Analysis</p>
<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>
<p>Item 4. Controls and Procedures</p>
<p>PART II OTHER INFORMATION</p>
<p>Item 1. Legal Proceedings</p>
<p>Item 1A. Risk Factors</p>
<p>Item 2. Unregistered Sales of Equity Securities</p>
<p>Item 6. Exhibits</p>
<p>PART I FINANCIAL INFORMATION</p>
<p>Item 1. Financial Statements</p>
<p>Condensed balance sheets and income statements follow.</p>
<p>Item 2. Management's Discussion and Analysis</p>
<p>Revenue grew 8 percent year over year on services strength.</p>
<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>
<p>No material change since the most recent 10-K.</p>
<p>Item 4. Controls and Procedures</p>
<p>Disclosure controls were effective.</p>
<p>PART II OTHER INFORMATION</p>
<p>Item 1. Legal Proceedings</p>
<p>See note 11.</p>
<p>Item 1A. Risk Factors</p>
<p>No material changes since the 10-K risk factors.</p>
<p>Item 2. Unregistered Sales of Equity Securities</p>
<p>The Company repurchased 100M shares during the quarter.</p>
<p>Item 6. Exhibits</p>
<p>Exhibit index follows.</p>
</body></html>"""


def test_html_to_text_strips_tags() -> None:
    text = html_to_text(_FAKE_10K)
    assert "<html>" not in text
    assert "Item 1. Business" in text
    assert "iPhones" in text


def test_find_item_offsets_returns_ordered() -> None:
    text = html_to_text(_FAKE_10K)
    offsets = find_item_offsets(text)
    tokens = [t for t, _, _ in offsets]
    # Order in the document is 1, 1A, 7, 8, 9; that order must be preserved.
    assert tokens == ["1", "1A", "7", "8", "9"]


def test_segments_into_sections_no_overlap() -> None:
    text = html_to_text(_FAKE_10K)
    sections = segment_into_sections(text)
    names = [s.name for s in sections]
    assert names == [
        ITEM_TO_CANONICAL["1"],
        ITEM_TO_CANONICAL["1A"],
        ITEM_TO_CANONICAL["7"],
        ITEM_TO_CANONICAL["8"],
        ITEM_TO_CANONICAL["9"],
    ]
    # No chunk should cross an Item boundary.
    for i in range(len(sections) - 1):
        assert sections[i].char_end <= sections[i + 1].char_start


def test_parse_filing_returns_sections() -> None:
    sections = parse_filing_html(_FAKE_10K)
    assert len(sections) == 5
    assert all(isinstance(s, Section) for s in sections)


def test_item_8_is_in_excluded_set() -> None:
    # Item 8 financial statements are tables of XBRL facts; they belong in
    # Postgres, not the vector index.
    assert "item_8_financial_statements" in SECTIONS_EXCLUDED_FROM_VECTOR_INDEX


def test_section_text_starts_at_item_header() -> None:
    sections = parse_filing_html(_FAKE_10K)
    business = next(s for s in sections if s.name == "item_1_business")
    assert business.text.lower().startswith("item 1")


def test_unknown_items_ignored() -> None:
    html = "<html><body><p>Item 99. Bogus</p><p>noise</p><p>Item 1. Business</p><p>real</p></body></html>"
    sections = parse_filing_html(html)
    names = [s.name for s in sections]
    assert "item_1_business" in names
    # No fictional item_99 emitted.
    assert all("item_99" not in n for n in names)


def test_10q_distinguishes_parts() -> None:
    # The collision case: 10-Q "Item 2" appears in both PART I (MD&A) and
    # PART II (Unregistered Sales). The fix must emit both sections under
    # their part-aware canonical names.
    sections = parse_filing_html(_FAKE_10Q, form="10-Q")
    names = [s.name for s in sections]
    assert "item_2_mda" in names
    assert "item_2_unregistered_sales" in names
    # Item 1 has the same collision (Financial Statements vs Legal Proceedings).
    assert "item_1_financial_statements" in names
    assert "item_1_legal_proceedings" in names
    # Risk factors maps to the same canonical name as 10-K Item 1A so the
    # axis filter works across forms.
    assert "item_1a_risk_factors" in names


def test_10q_section_content_matches_part() -> None:
    sections = parse_filing_html(_FAKE_10Q, form="10-Q")
    mda = next(s for s in sections if s.name == "item_2_mda")
    unreg = next(s for s in sections if s.name == "item_2_unregistered_sales")
    assert "revenue grew" in mda.text.lower()
    assert "repurchased" in unreg.text.lower()


def test_10q_find_item_offsets_attaches_part() -> None:
    text = html_to_text(_FAKE_10Q)
    offsets = find_item_offsets(text, form="10-Q")
    # Verify both Part-I Item 2 and Part-II Item 2 survived the dedup with
    # different part labels.
    item_2_entries = [(part, off) for tok, part, off in offsets if tok == "2"]
    parts = sorted(part for part, _ in item_2_entries)
    assert parts == ["1", "2"]


def test_10k_unaffected_by_part_logic() -> None:
    # 10-K parsing must produce the same five sections as before the fix.
    sections = parse_filing_html(_FAKE_10K, form="10-K")
    assert [s.name for s in sections] == [
        "item_1_business",
        "item_1a_risk_factors",
        "item_7_mda",
        "item_8_financial_statements",
        "item_9_changes_in_accountants",
    ]
