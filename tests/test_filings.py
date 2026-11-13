"""Filings stage: joined panel, links, SEC submissions and corpus facts in, filings table out."""

from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.corpus import load_corpus
from ai_mcdm_invest.data.filings import build_filings
from ai_mcdm_invest.data.joined import join_panel
from ai_mcdm_invest.data.links import link_companies, load_ticker_ciks, load_xbrl_figures
from ai_mcdm_invest.data.panel import clean_panel, load_raw_panel
from ai_mcdm_invest.data.submissions import load_submissions

FIXTURE_RAW_DIR = Path(__file__).parent / "fixtures" / "raw"


@pytest.fixture(scope="module")
def filings() -> pd.DataFrame:
    settings = Settings(raw_dir=FIXTURE_RAW_DIR)
    clean = clean_panel(load_raw_panel(settings.raw_panel_dir))
    links = link_companies(
        clean,
        load_xbrl_figures(settings.raw_frames_dir),
        load_ticker_ciks(settings.ticker_list_path),
    )
    joined = join_panel(clean, links)
    ciks = set(joined["cik"])
    return build_filings(
        joined,
        links,
        load_submissions(settings.submissions_path, ciks),
        load_corpus(settings.raw_corpus_dir, ciks),
    )


@pytest.fixture(scope="module")
def by_row(filings: pd.DataFrame) -> pd.DataFrame:
    return filings.set_index(["ticker", "fiscal_year"])


def test_one_row_per_joined_company_year(filings: pd.DataFrame) -> None:
    assert len(filings) == 22 * 5
    assert not filings.duplicated(["cik", "fiscal_year"]).any()
    assert list(filings.columns[:5]) == ["cik", "fiscal_year", "ticker", "sector", "link_method"]
    assert {"next_year_return", "tail_loss"} <= set(filings.columns)


def test_non_december_year_end_selects_the_filing_for_that_period(by_row: pd.DataFrame) -> None:
    # MIKE's fiscal 2017 ends 31 January 2017; the corpus labels that 10-K 2016.
    mike = by_row.loc[("MIKE", 2017)]

    assert mike["period_end"] == pd.Timestamp("2017-01-31")
    assert mike["filing_date"] == pd.Timestamp("2017-04-15")
    assert mike["corpus_year"] == 2016
    assert mike["days_after_fy_end"] == 74
    assert mike["days_after_jan1"] == -261  # filed before the return year (2018) began
    assert not mike["december_year_end"]


def test_december_filing_dates_and_lags(by_row: pd.DataFrame) -> None:
    alfa = by_row.loc[("ALFA", 2016)]  # also filed a 10-K/A, which is ignored

    assert alfa["filing_date"] == pd.Timestamp("2017-03-02")
    assert alfa["days_after_fy_end"] == 61
    assert alfa["days_after_jan1"] == 60
    assert alfa["december_year_end"]
    assert not alfa["nt_10k"]


def test_late_filing_notice_is_flagged(by_row: pd.DataFrame) -> None:
    oscr = by_row.loc[("OSCR", 2016)]

    assert oscr["nt_10k"]
    assert oscr["filing_date"] == pd.Timestamp("2017-04-14")
    assert not by_row.loc[("OSCR", 2017), "nt_10k"]


def test_ticker_linked_company_gets_the_10k_ending_in_its_fiscal_year(
    by_row: pd.DataFrame,
) -> None:
    fxtr = by_row.loc[("FXTR", 2015)]  # linked by ticker: no XBRL period

    assert fxtr["link_method"] == "ticker"
    assert fxtr["period_end"] == pd.Timestamp("2015-12-31")
    assert fxtr["filing_date"] == pd.Timestamp("2016-03-02")


def test_company_year_without_a_10k_keeps_its_row_with_empty_filing_columns(
    by_row: pd.DataFrame,
) -> None:
    qubc = by_row.loc[("QUBC", 2017)]

    assert pd.isna(qubc["filing_date"])
    assert pd.isna(qubc["days_after_jan1"])
    assert not qubc["in_corpus"]
    assert pd.isna(qubc["has_item_1A"])
    assert qubc["tail_loss"]  # still a tail loser, counted by the funnel


def test_10k_missing_from_the_corpus_has_a_filing_date_but_no_text(by_row: pd.DataFrame) -> None:
    papa = by_row.loc[("PAPA", 2015)]

    assert papa["filing_date"] == pd.Timestamp("2016-03-02")
    assert not papa["in_corpus"]
    assert pd.isna(papa["tokens_total"])


def test_section_facts_come_from_the_selected_10k(by_row: pd.DataFrame) -> None:
    novm = by_row.loc[("NOVM", 2016)]
    assert novm["in_corpus"]
    assert not novm["has_item_1A"]  # "Not applicable."
    assert not novm["has_item_8"]  # points to Item 15
    assert novm["tokens_item_1"] == 500
    assert by_row.loc[("ROMO", 2015), "gc_keyword_items"] == "1A;8"


def test_item_1a_change_share_against_the_prior_year(by_row: pd.DataFrame) -> None:
    # KILO 2016: of four paragraphs one is new; another changed one sentence of three.
    assert by_row.loc[("KILO", 2016), "item1a_changed_share"] == 0.25
    assert by_row.loc[("KILO", 2017), "item1a_changed_share"] == 0.0
    # Fiscal 2014 compares with the 10-K for the period ending in 2013.
    assert by_row.loc[("ALFA", 2014), "item1a_changed_share"] == 0.0


def test_missing_prior_year_leaves_the_change_share_empty(by_row: pd.DataFrame) -> None:
    # PAPA's fiscal 2015 10-K is not in the corpus.
    assert pd.isna(by_row.loc[("PAPA", 2016), "item1a_changed_share"])
    assert by_row.loc[("PAPA", 2017), "item1a_changed_share"] == 0.0


def test_stub_item_1a_has_no_change_share(by_row: pd.DataFrame) -> None:
    assert pd.isna(by_row.loc[("NOVM", 2016), "item1a_changed_share"])
    # and a stub prior year counts as missing
    assert pd.isna(by_row.loc[("NOVM", 2017), "item1a_changed_share"])
