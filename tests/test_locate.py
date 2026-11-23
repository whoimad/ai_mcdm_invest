"""Locating the auditor's report: a 10-K's Item texts in, the chunks the LLM reads out."""

from ai_mcdm_invest.llm.locate import CHUNK_CHARS, locate_auditor_reports

HEADING = "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM"
OPINION = (
    "To the Board of Directors and Stockholders of Alfa Corp.\n"
    "We have audited the accompanying consolidated balance sheets of Alfa Corp. as of "
    "December 31, 2015 and 2014. In our opinion, the financial statements present fairly, "
    "in all material respects, the financial position of the Company.\n"
)
SIGNATURE = "/s/ Smith & Jones LLP\nDenver, Colorado\nMarch 14, 2016\n"
NOTES = "CONSOLIDATED BALANCE SHEETS\nNote 1. Organization. The Company makes widgets.\n"


def report(opinion: str = OPINION) -> str:
    return f"{HEADING}\n{opinion}{SIGNATURE}"


def test_report_in_item_8_is_cut_from_heading_to_signature_date() -> None:
    item_8 = f"Item 8. Financial Statements\n{report()}{NOTES}"

    located = locate_auditor_reports({"8": item_8, "15": "Item 15. Exhibits\n"})

    assert not located.fallback_read
    assert [chunk.item for chunk in located.chunks] == ["8"]
    assert located.chunks[0].text == report().rstrip("\n")


def test_report_only_in_item_15_is_found_there() -> None:
    item_8 = "Item 8. The financial statements are set forth in Item 15.\n"
    item_15 = f"Item 15. Exhibits, Financial Statement Schedules\n{report()}{NOTES}"

    located = locate_auditor_reports({"8": item_8, "15": item_15})

    assert not located.fallback_read
    assert [chunk.item for chunk in located.chunks] == ["15"]
    assert located.chunks[0].text == report().rstrip("\n")


def test_two_reports_are_two_chunks_and_index_mentions_are_skipped() -> None:
    icfr_opinion = (
        "To the Board of Directors of Alfa Corp.\n"
        "We have audited Alfa Corp.'s internal control over financial reporting as of "
        "December 31, 2015. In our opinion, the Company maintained effective internal control.\n"
    )
    index = f"Index to Financial Statements\n{HEADING} F-1\n{HEADING} F-2\nBalance Sheets F-3\n"
    item_8 = f"Item 8.\n{index}{report()}{report(icfr_opinion)}{NOTES}"

    located = locate_auditor_reports({"8": item_8})

    assert [chunk.text for chunk in located.chunks] == [
        report().rstrip("\n"),
        report(icfr_opinion).rstrip("\n"),
    ]


def test_over_long_report_is_capped_at_about_6k_tokens() -> None:
    padding = "The Company restated its segment disclosures for 2014. " * 1000  # ~55k chars
    item_8 = f"Item 8.\n{report(OPINION + padding + chr(10))}{NOTES}"

    (chunk,) = locate_auditor_reports({"8": item_8}).chunks

    assert len(chunk.text) == CHUNK_CHARS == 24_000
    assert chunk.text.startswith(HEADING)


def test_without_a_heading_both_items_are_read_in_full_in_6k_chunks() -> None:
    line = "The Company's financial statements were audited by Smith & Jones LLP.\n"
    item_8 = "Item 8.\n" + line * 500  # ~35k chars: two chunks
    item_15 = "Item 15.\n" + line * 10

    located = locate_auditor_reports({"8": item_8, "15": item_15})

    assert located.fallback_read
    assert [chunk.item for chunk in located.chunks] == ["8", "8", "15"]
    assert all(len(chunk.text) <= CHUNK_CHARS for chunk in located.chunks)
    assert "".join(chunk.text for chunk in located.chunks) == item_8 + item_15
    assert located.chunks[0].text.endswith("\n")  # cut between lines, not mid-sentence


def test_a_heading_without_an_audit_opinion_is_a_reference_not_a_report() -> None:
    item_15 = f"Item 15.\nExhibit 34.1 {HEADING} (Smith & Jones LLP).\nSIGNATURES\n"

    located = locate_auditor_reports({"8": "", "15": item_15})

    assert located.fallback_read
    assert [chunk.text for chunk in located.chunks] == [item_15]


def test_a_10k_without_items_8_and_15_has_nothing_to_read() -> None:
    located = locate_auditor_reports({"1": "Item 1. Business."})

    assert located.fallback_read
    assert located.chunks == ()


def test_a_line_opening_with_a_balance_sheet_date_does_not_end_the_report() -> None:
    wrapped = (
        "We have audited the accompanying balance sheets of Alfa Corp. as of\n"
        "December 31, 2015 and 2014, and the related statements of operations.\n"
        "These factors raise substantial doubt about its ability to continue as a going concern.\n"
    )

    (chunk,) = locate_auditor_reports({"8": f"Item 8.\n{report(wrapped)}{NOTES}"}).chunks

    assert chunk.text == report(wrapped).rstrip("\n")


def test_a_signature_date_after_the_city_on_one_line_ends_the_report() -> None:
    signed = f"{HEADING}\n{OPINION}/s/ Smith & Jones LLP\nDenver, Colorado, March 14, 2016\n"

    (chunk,) = locate_auditor_reports({"8": f"Item 8.\n{signed}{NOTES}"}).chunks

    assert chunk.text == signed.rstrip("\n")


def test_a_fallback_line_longer_than_a_chunk_is_cut_at_the_cap() -> None:
    item_8 = "x" * (2 * CHUNK_CHARS + 100)

    located = locate_auditor_reports({"8": item_8})

    assert [len(chunk.text) for chunk in located.chunks] == [CHUNK_CHARS, CHUNK_CHARS, 100]
