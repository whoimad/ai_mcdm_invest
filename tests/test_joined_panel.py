"""Joined-panel stage: clean panel and company links in, one row per filer and year out."""

import pandas as pd

from ai_mcdm_invest.data.joined import join_panel


def test_share_classes_of_one_filer_collapse_to_the_largest(
    clean_fixture_panel: pd.DataFrame, fixture_links: pd.DataFrame
) -> None:
    # LIMA, its class K (twice the market cap) and its preferred LIMAP all
    # report LIMA's figures in 2018, so all three link to LIMA's filer.
    joined = join_panel(clean_fixture_panel, fixture_links.reset_index())

    lima = joined[(joined["cik"] == 1011) & (joined["fiscal_year"] == 2018)]
    assert lima["ticker"].tolist() == ["LIMAK"]
    assert lima["Market Cap"].item() == 72240.0
    assert lima["next_year_return"].item() == 40.0


def join_rows(*rows: tuple[str, float | None, str]) -> pd.DataFrame:
    """Join 2017 panel rows (ticker, market cap, link method), all linked to CIK 42."""
    tickers = [ticker for ticker, _, _ in rows]
    clean = pd.DataFrame(
        {
            "ticker": tickers,
            "fiscal_year": 2017,
            "sector": "Financial Services",
            "Market Cap": [cap for _, cap, _ in rows],
            "next_year_return": 5.0,
            "tail_loss": False,
        }
    )
    links = pd.DataFrame(
        {
            "ticker": tickers,
            "fiscal_year": 2017,
            "cik": pd.array([42] * len(rows), dtype="Int64"),
            "link_method": [method for _, _, method in rows],
        }
    )
    return join_panel(clean, links)


def test_reused_ticker_loses_to_the_row_the_figures_matched() -> None:
    # Today's BBT ticker points at the filer BHLB's own figures matched.
    joined = join_rows(("BBT", 28e9, "ticker"), ("BHLB", 0.7e9, "figure_match"))

    assert joined["ticker"].tolist() == ["BHLB"]


def test_preferred_share_without_a_market_cap_loses_even_when_matched_by_figures() -> None:
    joined = join_rows(("AGNC", 7.7e9, "ticker"), ("AGNCB", None, "figure_match"))

    assert joined["ticker"].tolist() == ["AGNC"]


def test_shortest_ticker_is_kept_when_no_row_has_a_market_cap() -> None:
    joined = join_rows(("OXLCM", None, "ticker"), ("OXLC", None, "ticker"))

    assert joined["ticker"].tolist() == ["OXLC"]


def test_joined_panel_has_one_row_per_filer_and_year_and_leaves_out_unlinked_rows(
    clean_fixture_panel: pd.DataFrame, fixture_links: pd.DataFrame
) -> None:
    joined = join_panel(clean_fixture_panel, fixture_links.reset_index())

    assert not joined.duplicated(["cik", "fiscal_year"]).any()
    assert not {"CHRL", "DLTA", "HOTL"} & set(joined["ticker"])  # unlinked
    assert len(joined) == 22 * 5  # 25 core companies minus the three unlinked
    assert list(joined.columns[:4]) == ["cik", "fiscal_year", "ticker", "link_method"]
    assert {"sector", "Total assets", "next_year_return", "tail_loss"} <= set(joined.columns)
    alfa = joined.set_index(["ticker", "fiscal_year"]).loc[("ALFA", 2017)]
    assert alfa["cik"] == 1000  # the figure match, not the reused ticker's 9001
    assert alfa["sector"] == "Technology"
