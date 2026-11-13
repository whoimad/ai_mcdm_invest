"""Company-link stage: clean panel, XBRL figures and ticker list in, links out."""

import pandas as pd

from ai_mcdm_invest.data.links import link_companies


def test_figure_match_wins_over_a_reused_ticker(fixture_links: pd.DataFrame) -> None:
    # Today's ticker list gives ALFA to CIK 9001; ALFA's own figures are CIK 1000's.
    for year in range(2014, 2019):
        alfa = fixture_links.loc[("ALFA", year)]
        assert alfa["ticker_cik"] == 9001
        assert alfa["cik"] == 1000
        assert alfa["link_method"] == "figure_match"


def test_company_missing_from_the_ticker_list_is_recovered_by_figures(
    fixture_links: pd.DataFrame,
) -> None:
    brvo = fixture_links.loc[("BRVO", 2017)]

    assert pd.isna(brvo["ticker_cik"])
    assert brvo["cik"] == 1001
    assert brvo["link_method"] == "figure_match"
    assert brvo["filer_name"] == "BRVO Corp"
    assert brvo["period_end"] == "2017-12-31"
    assert brvo["matched_revenue"]
    assert brvo["matched_net_income"]


def test_match_to_two_filers_is_rejected(fixture_links: pd.DataFrame) -> None:
    chrl = fixture_links.loc[("CHRL", 2017)]

    assert chrl["figure_match"] == "ambiguous"
    assert pd.isna(chrl["figure_cik"])
    assert pd.isna(chrl["cik"])
    assert chrl["link_method"] == "unlinked"


def test_ambiguous_match_that_includes_the_ticker_filer_keeps_the_ticker_link(
    fixture_links: pd.DataFrame,
) -> None:
    golf = fixture_links.loc[("GOLF", 2017)]  # figures match CIK 1006 and 9003

    assert golf["figure_match"] == "ambiguous"
    assert golf["cik"] == 1006
    assert golf["link_method"] == "ticker"


def test_ambiguous_match_that_excludes_the_ticker_filer_leaves_the_row_unlinked(
    fixture_links: pd.DataFrame,
) -> None:
    hotl = fixture_links.loc[("HOTL", 2017)]  # ticker says 9004; figures say 1007 or 9005

    assert hotl["ticker_cik"] == 9004
    assert hotl["figure_match"] == "ambiguous"
    assert pd.isna(hotl["cik"])
    assert hotl["link_method"] == "unlinked"


def test_company_found_by_neither_method_is_unlinked(fixture_links: pd.DataFrame) -> None:
    dlta = fixture_links.loc[("DLTA", 2017)]

    assert dlta["figure_match"] == "none"
    assert pd.isna(dlta["cik"])
    assert dlta["link_method"] == "unlinked"


def test_ticker_link_is_kept_when_figures_find_no_filer(fixture_links: pd.DataFrame) -> None:
    fxtr = fixture_links.loc[("FXTR", 2017)]

    assert fxtr["figure_match"] == "none"
    assert fxtr["cik"] == 1005
    assert fxtr["link_method"] == "ticker"


def test_net_income_alone_confirms_the_match_when_revenue_is_missing(
    fixture_links: pd.DataFrame,
) -> None:
    echo = fixture_links.loc[("ECHO", 2014)]  # no revenue in the panel

    assert echo["link_method"] == "figure_match"
    assert echo["cik"] == 1004
    assert not echo["matched_revenue"]
    assert echo["matched_net_income"]


def test_every_panel_row_gets_exactly_one_link_row(
    clean_fixture_panel: pd.DataFrame, fixture_links: pd.DataFrame
) -> None:
    assert fixture_links.index.is_unique
    assert len(fixture_links) == len(clean_fixture_panel) == 127
    assert fixture_links["link_method"].value_counts().to_dict() == {
        "figure_match": 20 * 5 + 2,  # 25 core companies minus the five below, plus LIMAK, LIMAP
        "unlinked": 3 * 5,  # CHRL, DLTA, HOTL
        "ticker": 2 * 5,  # FXTR, GOLF
    }


def link_one(assets: float, revenue: float, period_end: str = "2017-12-31") -> pd.Series:
    """Link one 2017 panel row (assets 1,000,000, revenue 500,000) to one filer."""
    panel = pd.DataFrame(
        {
            "ticker": ["ONLY"],
            "fiscal_year": [2017],
            "Total assets": [1_000_000.0],
            "Revenue": [500_000.0],
            "Net Income": [float("nan")],
        }
    )
    xbrl = pd.DataFrame(
        {
            "cik": [42, 42],
            "filer_name": ["Only Inc", "Only Inc"],
            "period_end": [period_end, period_end],
            "figure": ["total_assets", "revenue"],
            "value": [assets, revenue],
        }
    )
    no_tickers = pd.DataFrame({"ticker": pd.Series([], dtype=str), "cik": pd.Series([], dtype=int)})
    return link_companies(panel, xbrl, no_tickers).iloc[0]


def test_figures_within_a_tenth_of_a_percent_match() -> None:
    assert link_one(assets=1_000_900.0, revenue=499_600.0)["cik"] == 42


def test_figures_off_by_more_than_a_tenth_of_a_percent_do_not_match() -> None:
    assert link_one(assets=1_002_000.0, revenue=500_000.0)["figure_match"] == "none"
    assert link_one(assets=1_000_000.0, revenue=501_000.0)["figure_match"] == "none"


def test_period_ending_in_another_fiscal_year_is_not_a_candidate() -> None:
    assert link_one(1_000_000.0, 500_000.0, period_end="2016-12-31")["figure_match"] == "none"
    assert link_one(1_000_000.0, 500_000.0, period_end="2017-01-28")["cik"] == 42
