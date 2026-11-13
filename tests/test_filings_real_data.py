"""Slow acceptance check on the real raw data: the filings table finds the 10-Ks.

Kept out of CI (it needs the raw data, about 16 GB); run with `make test-slow`.
"""

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.corpus import load_corpus
from ai_mcdm_invest.data.filings import build_filings
from ai_mcdm_invest.data.joined import join_panel
from ai_mcdm_invest.data.links import link_companies, load_ticker_ciks, load_xbrl_figures
from ai_mcdm_invest.data.panel import clean_panel, load_raw_panel
from ai_mcdm_invest.data.raw_manifest import load_manifest, verify_raw_files
from ai_mcdm_invest.data.submissions import load_submissions

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def filings() -> pd.DataFrame:
    settings = Settings.from_env()
    if not settings.raw_corpus_dir.is_dir():
        pytest.skip(f"real raw data not found in {settings.raw_dir}")
    verify_raw_files(settings.raw_dir, load_manifest(settings.manifest_path))
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


def test_fiscal_2017_coverage(filings: pd.DataFrame) -> None:
    # Measured on the SEC data of 13 November 2026: of the joined rows, 81.5%
    # have a 10-K for their period and 79.5% its text in the corpus. Most of
    # the rest are foreign filers (20-F, 40-F), funds, and companies whose
    # first 10-K came a year later.
    year = filings[filings["fiscal_year"] == 2017]
    assert year["filing_date"].notna().mean() == pytest.approx(0.815, abs=0.01)
    assert year["in_corpus"].mean() == pytest.approx(0.795, abs=0.01)
    assert year.loc[year["tail_loss"], "in_corpus"].mean() > 0.8


def test_known_filings_are_chosen_by_period(filings: pd.DataFrame) -> None:
    rows = filings.set_index(["ticker", "fiscal_year"])
    # Walmart's fiscal year ends 31 January.
    walmart = rows.loc[("WMT", 2018)]
    assert walmart["period_end"] == pd.Timestamp("2018-01-31")
    assert walmart["filing_date"] == pd.Timestamp("2018-03-30")
    # The corpus labels Amazon's fiscal 2018 10-K 2019, the year it was filed.
    amazon = rows.loc[("AMZN", 2018)]
    assert amazon["filing_date"] == pd.Timestamp("2019-02-01")
    assert amazon["corpus_year"] == 2019
