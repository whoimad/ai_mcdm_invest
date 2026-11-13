"""Cleaning stage: raw fundamentals panel in, clean panel out."""

import pandas as pd
import pytest

from ai_mcdm_invest.data.panel import clean_panel


def test_one_row_per_ticker_and_fiscal_year(raw_panel: pd.DataFrame) -> None:
    clean = clean_panel(raw_panel)

    assert not clean.duplicated(["ticker", "fiscal_year"]).any()
    assert sorted(clean["fiscal_year"].unique()) == [2014, 2015, 2016, 2017, 2018]
    # 25 core companies, plus LIMA's two other securities in 2018.
    assert clean.groupby("fiscal_year")["ticker"].nunique().to_dict() == {
        2014: 25,
        2015: 25,
        2016: 25,
        2017: 25,
        2018: 27,
    }


def test_rows_that_cannot_be_labelled_are_dropped(raw_panel: pd.DataFrame) -> None:
    raw_tickers = set(raw_panel["ticker"])
    assert {"ZULU", "ZZNR", "ZZWO"} <= raw_tickers  # present before cleaning

    clean = clean_panel(raw_panel)

    # ZULU: +2500% (data error). ZZNR: no return. ZZWO: -100%.
    assert not {"ZULU", "ZZNR", "ZZWO"} & set(clean["ticker"])
    assert len(clean) == 127


def test_keeps_fundamentals_sector_and_return(raw_panel: pd.DataFrame) -> None:
    clean = clean_panel(raw_panel).set_index(["ticker", "fiscal_year"])

    assert list(clean.columns) == [
        "sector",
        "Revenue",
        "Net Income",
        "Total assets",
        "Total debt",
        "Market Cap",
        "debtEquityRatio",
        "currentRatio",
        "returnOnEquity",
        "netProfitMargin",
        "priceEarningsRatio",
        "priceToBookRatio",
        "interestCoverage",
        "next_year_return",
        "tail_loss",
    ]
    brvo = clean.loc[("BRVO", 2014)]
    assert brvo["sector"] == "Energy"
    assert brvo["Revenue"] == 2000.0
    assert brvo["Total assets"] == 4000.0
    assert brvo["next_year_return"] == -62.5


def test_row_with_a_missing_fundamental_is_kept(raw_panel: pd.DataFrame) -> None:
    clean = clean_panel(raw_panel).set_index(["ticker", "fiscal_year"])

    assert pd.isna(clean.loc[("ECHO", 2014), "Revenue"])
    assert clean.loc[("ECHO", 2014), "Net Income"] == 500.0


def test_outcome_derived_class_column_is_removed(raw_panel: pd.DataFrame) -> None:
    assert "Class" in raw_panel.columns

    assert "Class" not in clean_panel(raw_panel).columns


def test_repeated_ticker_within_a_year_is_rejected(raw_panel: pd.DataFrame) -> None:
    repeated = pd.concat([raw_panel, raw_panel.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="ALFA"):
        clean_panel(repeated)


def tiny_raw_panel(returns: list[float]) -> pd.DataFrame:
    """One fiscal year of hand-built raw rows, one per return."""
    return pd.DataFrame(
        {
            "ticker": [f"T{i:02d}" for i in range(len(returns))],
            "Revenue": 1.0,
            "Sector": "Energy",
            "next_year_return": returns,
            "Class": 0,
            "fiscal_year": 2014,
        }
    )


def test_return_of_exactly_plus_1000_percent_is_kept_and_minus_100_dropped() -> None:
    clean = clean_panel(tiny_raw_panel([1000.0, 1000.01, -100.0, -99.99, 5.0]))

    assert sorted(clean["next_year_return"]) == [-99.99, 5.0, 1000.0]


def test_company_exactly_at_the_cut_off_is_a_tail_loser() -> None:
    # Six returns: the 20th percentile falls exactly on the second lowest (-40).
    clean = clean_panel(tiny_raw_panel([-50.0, -40.0, -30.0, -20.0, -10.0, 0.0]))

    assert clean.loc[clean["tail_loss"], "next_year_return"].tolist() == [-50.0, -40.0]


def test_year_whose_return_column_is_absent_is_rejected(raw_panel: pd.DataFrame) -> None:
    broken = raw_panel.copy()
    broken.loc[broken["fiscal_year"] == 2016, "next_year_return"] = float("nan")

    with pytest.raises(ValueError, match="2016"):
        clean_panel(broken)
