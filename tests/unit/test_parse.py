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

# Filler long enough to break the chained-header run so body sections read as real
# bodies, not table-of-contents lines.
_Q_FILLER = " ".join(["Quarterly narrative filler sentence past the body-gap threshold."] * 11)

# Realistic 10-Q: a packed table of contents (stripped as a navigation block) then
# real sections with substantial bodies, across both PARTs.
_FAKE_10Q = (
    "<html><body><h1>Quarterly Report</h1><p>Table of Contents</p>"
    "<p>PART I FINANCIAL INFORMATION</p>"
    "<p>Item 1. Financial Statements</p>"
    "<p>Item 2. Management's Discussion and Analysis</p>"
    "<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>"
    "<p>Item 4. Controls and Procedures</p>"
    "<p>PART II OTHER INFORMATION</p>"
    "<p>Item 1. Legal Proceedings</p>"
    "<p>Item 1A. Risk Factors</p>"
    "<p>Item 2. Unregistered Sales of Equity Securities</p>"
    "<p>Item 6. Exhibits</p>"
    "<p>PART I FINANCIAL INFORMATION</p>"
    f"<p>Item 1. Financial Statements</p><p>Condensed balance sheets follow. {_Q_FILLER}</p>"
    f"<p>Item 2. Management's Discussion and Analysis</p>"
    f"<p>Revenue grew 8 percent year over year on services strength. {_Q_FILLER}</p>"
    f"<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>"
    f"<p>No material change since the most recent 10-K. {_Q_FILLER}</p>"
    f"<p>Item 4. Controls and Procedures</p><p>Disclosure controls were effective. {_Q_FILLER}</p>"
    "<p>PART II OTHER INFORMATION</p>"
    f"<p>Item 1. Legal Proceedings</p><p>See note 11. {_Q_FILLER}</p>"
    f"<p>Item 1A. Risk Factors</p><p>No material changes since the 10-K risk factors. {_Q_FILLER}</p>"
    f"<p>Item 2. Unregistered Sales of Equity Securities</p>"
    f"<p>The Company repurchased 100M shares during the quarter. {_Q_FILLER}</p>"
    f"<p>Item 6. Exhibits</p><p>Exhibit index follows. {_Q_FILLER}</p>"
    "</body></html>"
)


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


_FILLER = " ".join(["Narrative padding sentence past the body-gap threshold."] * 11)

# A packed table of contents (every item) followed by real ALL-CAPS section
# headers, plus an in-body Title-Case cross-reference to Item 1A inside the
# Business section. Reproduces the PFE/MSFT 10-K shape that broke
# last-occurrence-wins.
_PATHO_10K = (
    "<html><body><p>Table of Contents</p>"
    + "".join(
        f"<p>Item {tok}. {name}</p>"
        for tok, name in [
            ("1", "Business"),
            ("1A", "Risk Factors"),
            ("1B", "Unresolved Staff Comments"),
            ("2", "Properties"),
            ("3", "Legal Proceedings"),
            ("4", "Mine Safety Disclosures"),
            ("5", "Market for Common Equity"),
            ("6", "Reserved"),
            ("7", "MD and A"),
            ("7A", "Market Risk"),
            ("8", "Financial Statements"),
            ("9", "Changes in Accountants"),
            ("9A", "Controls"),
            ("9B", "Other Information"),
            ("9C", "Foreign Jurisdictions"),
            ("10", "Directors"),
            ("11", "Compensation"),
            ("12", "Security Ownership"),
            ("13", "Related Transactions"),
            ("14", "Accountant Fees"),
            ("15", "Exhibits"),
        ]
    )
    + f"<p>Item 1. BUSINESS</p><p>We make widgets and provide services. {_FILLER}</p>"
    + "<p>Our reliance on suppliers is discussed in Item 1A. Risk Factors below.</p>"
    + f"<p>Item 1A. RISK FACTORS</p><p>This section describes the material risks to our business. {_FILLER}</p>"
    + f"<p>Item 7. MANAGEMENT DISCUSSION AND ANALYSIS</p><p>Operating margin improved. {_FILLER}</p>"
    + f"<p>Item 8. FINANCIAL STATEMENTS</p><p>Revenue and net income tables. {_FILLER}</p>"
    + f"<p>Item 9. CHANGES IN ACCOUNTANTS</p><p>None reported this year. {_FILLER}</p>"
    + "</body></html>"
)


def test_10k_toc_and_cross_references_do_not_scramble_sections() -> None:
    # Regression for the PFE/MSFT scramble: the TOC and an in-body cross-reference
    # must not become section anchors, so item_1a keeps its real body instead of
    # starving to a sliver and TOC-only items do not swallow following text.
    sections = {s.name: s.text for s in parse_filing_html(_PATHO_10K, form="10-K")}

    # Risk factors capture the real body, not the Title-Case cross-reference.
    assert "This section describes the material risks" in sections["item_1a_risk_factors"]
    assert len(sections["item_1a_risk_factors"]) > 400

    # Business keeps its real text and is not truncated by the cross-reference.
    assert "We make widgets" in sections["item_1_business"]

    # Items present only in the TOC (no body) are dropped, not anchored.
    for toc_only in (
        "item_4_mine_safety",
        "item_1b_unresolved_staff_comments",
        "item_9c_foreign_jurisdictions",
        "item_7a_market_risk",
    ):
        assert toc_only not in sections

    # No section runs away with the whole document.
    doc_len = len(html_to_text(_PATHO_10K))
    assert all(len(t) < doc_len * 0.6 for t in sections.values())


# A 10-Q whose TOC lists mine-safety and defaults (both N/A, absent from the body),
# with in-body cross-references inside the MD&A. Reproduces the PFE shape where
# last-occurrence-wins anchored mine_safety to its TOC entry and let it swallow the
# entire financial-statements block.
_PATHO_10Q = (
    "<html><body><p>Table of Contents</p>"
    "<p>PART I FINANCIAL INFORMATION</p>"
    "<p>Item 1. Financial Statements</p>"
    "<p>Item 2. Management's Discussion and Analysis</p>"
    "<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>"
    "<p>Item 4. Controls and Procedures</p>"
    "<p>PART II OTHER INFORMATION</p>"
    "<p>Item 1. Legal Proceedings</p>"
    "<p>Item 1A. Risk Factors</p>"
    "<p>Item 2. Unregistered Sales of Equity Securities</p>"
    "<p>Item 3. Defaults Upon Senior Securities N/A</p>"
    "<p>Item 4. Mine Safety Disclosures N/A</p>"
    "<p>Item 5. Other Information</p>"
    "<p>Item 6. Exhibits</p>"
    "<p>PART I FINANCIAL INFORMATION</p>"
    f"<p>Item 1. Financial Statements</p><p>Condensed consolidated balance sheets. {_Q_FILLER}</p>"
    f"<p>Item 2. Management's Discussion and Analysis</p>"
    f"<p>Revenue grew on volume. Risks are discussed in Item 1A. Risk Factors below. {_Q_FILLER}</p>"
    f"<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p><p>No change. {_Q_FILLER}</p>"
    f"<p>Item 4. Controls and Procedures</p><p>Effective. {_Q_FILLER}</p>"
    "<p>PART II OTHER INFORMATION</p>"
    "<p>Item 1. Legal Proceedings</p><p>See Note 12.</p>"
    "<p>Item 1A. Risk Factors</p><p>No material changes since the 10-K.</p>"
    f"<p>Item 2. Unregistered Sales of Equity Securities</p><p>Repurchased shares. {_Q_FILLER}</p>"
    "<p>Item 5. Other Information</p><p>None.</p>"
    f"<p>Item 6. Exhibits</p><p>Exhibit index. {_Q_FILLER}</p>"
    "</body></html>"
)


def test_10q_absent_items_do_not_swallow_body() -> None:
    # Regression for the PFE 10-Q scramble: mine safety and defaults are listed in
    # the TOC but N/A in the body, so they must be dropped, not anchored to their
    # TOC entry where they would swallow the financial-statements block.
    sections = {s.name: s.text for s in parse_filing_html(_PATHO_10Q, form="10-Q")}

    for absent in ("item_4_mine_safety", "item_3_defaults_upon_senior_securities"):
        assert absent not in sections

    # MD&A keeps its real body and is not truncated by the in-body cross-reference.
    assert "Revenue grew on volume" in sections["item_2_mda"]
    assert len(sections["item_2_mda"]) > 400

    # Financial statements are captured (the block mine_safety used to swallow).
    assert "Condensed consolidated balance sheets" in sections["item_1_financial_statements"]

    # No section runs away with the whole document.
    doc_len = len(html_to_text(_PATHO_10Q))
    assert all(len(t) < doc_len * 0.6 for t in sections.values())
