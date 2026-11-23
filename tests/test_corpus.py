"""EDGAR-CORPUS loader: 10-K records split by Item in, facts about each record out."""

from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.corpus import describe_record, load_corpus


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    raw_dir = Path(__file__).parent / "fixtures" / "raw"
    return load_corpus(Settings(raw_dir=raw_dir).raw_corpus_dir, {1000, 1012, 1013, 1017})


def record(corpus: pd.DataFrame, cik: int, period_end: str) -> pd.Series:
    return corpus.set_index(["cik", "period_end"]).loc[(cik, pd.Timestamp(period_end))]


def test_keeps_only_the_requested_filers(corpus: pd.DataFrame) -> None:
    assert set(corpus["cik"]) == {1000, 1012, 1013, 1017}
    assert len(corpus) == 4 * 6  # one 10-K a year, periods ending 2013-2018


def test_period_comes_from_the_text_not_the_label(corpus: pd.DataFrame) -> None:
    # MIKE's year ending January 2017 is labelled 2016 in the corpus.
    mike = record(corpus, 1012, "2017-01-31")

    assert mike["corpus_year"] == 2016


def test_tokens_are_characters_over_four(corpus: pd.DataFrame) -> None:
    alfa = record(corpus, 1000, "2016-12-31")

    assert alfa["tokens_item_1"] == 500  # Item 1 is exactly 2,000 characters
    assert alfa["tokens_item_1A"] > 0
    assert alfa["tokens_total"] > alfa["tokens_item_1"] + alfa["tokens_item_1A"]


def test_stub_sections_count_as_absent(corpus: pd.DataFrame) -> None:
    # NOVM 2016: Item 1A reads "Not applicable." and Item 8 points to Item 15.
    novm = record(corpus, 1013, "2016-12-31")

    assert not novm["has_item_1A"]
    assert not novm["has_item_8"]
    assert novm["has_item_7"]
    assert novm["has_item_3"]  # short but real: "none of which is material"
    assert record(corpus, 1013, "2015-12-31")["has_item_1A"]


def test_going_concern_wording_is_located_by_item(corpus: pd.DataFrame) -> None:
    assert record(corpus, 1017, "2015-12-31")["gc_keyword_items"] == "1A;8"
    assert record(corpus, 1017, "2016-12-31")["gc_keyword_items"] == ""


def test_a_10k_without_a_located_auditors_report_is_marked_fallback_read() -> None:
    report = (
        "Report of Independent Registered Public Accounting Firm\n"
        "We have audited the balance sheets of Alfa Corp. as of December 31, 2015.\n"
        "/s/ Smith & Jones LLP\nMarch 14, 2016\n"
    )
    with_report = {"cik": "1000", "year": "2015", "section_8": report, "section_15": ""}
    without = {"cik": "1000", "year": "2015", "section_8": "Item 8. See Item 15.", "section_15": ""}

    assert not describe_record(with_report)["fallback_read"]
    assert describe_record(without)["fallback_read"]


def test_a_10k_without_any_audit_opinion_text_is_marked_no_report() -> None:
    unheaded = "Item 8.\nWe have audited the balance sheets of Alfa Corp.\n"
    with_text = {"cik": "1000", "year": "2015", "section_8": unheaded, "section_15": ""}
    without = {
        "cik": "1000",
        "year": "2015",
        "section_8": "Item 8. See Exhibit 13.",
        "section_15": "",
    }

    assert not describe_record(with_text)["no_report"]
    assert describe_record(without)["no_report"]


def test_missing_corpus_folder_says_how_to_fetch(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make fetch"):
        load_corpus(tmp_path / "nowhere", {1000})
