"""Evaluation: the baselines and the layer ablation table."""

import pandas as pd
import pytest

from ai_mcdm_invest.evaluation.ablation import (
    ablation_table,
    baselines,
    evaluation_universe,
    ranking_metrics,
)


def test_last_year_s_price_move_is_the_return_the_prior_fiscal_year_looked_ahead_to() -> None:
    # Fiscal 2016's next-year return is the price move over 2017, the year
    # fiscal 2017 ends in: known before fiscal 2017's own return year begins.
    joined = pd.DataFrame(
        {
            "cik": [1, 1, 2, 3],
            "fiscal_year": [2016, 2017, 2017, 2015],
            "ticker": ["A", "A", "B", "C"],
            "Market Cap": [5e9, 6e9, None, 1e8],
            "next_year_return": [12.0, -40.0, 3.0, 7.0],
        }
    )

    table = baselines(joined).set_index(["cik", "fiscal_year"])

    assert table.loc[(1, 2017), "last_year_return"] == 12.0
    assert pd.isna(table.loc[(1, 2016), "last_year_return"])  # no fiscal 2015 row
    assert pd.isna(table.loc[(2, 2017), "last_year_return"])
    assert table.loc[(1, 2017), "market_cap"] == 6e9
    assert pd.isna(table.loc[(2, 2017), "market_cap"])


def one_year(fiscal_year: int = 2017, shift: float = 0.0) -> pd.DataFrame:
    """Ten companies scored 1 to 10 (higher safer); the two worst returns are
    the tail losers, scored 1 and 3."""
    returns = [-50.0, -10.0, -30.0, 5.0, 0.0, 10.0, 8.0, 20.0, 15.0, 30.0]
    return pd.DataFrame(
        {
            "fiscal_year": fiscal_year,
            "score": [shift + i for i in range(1, 11)],
            "next_year_return": returns,
            "tail_loss": [r <= -30 for r in returns],
        }
    )


def test_a_ranking_is_described_by_auc_fifths_rank_correlation_and_return_spread() -> None:
    metrics = ranking_metrics(one_year())

    assert metrics["companies"] == 10
    assert metrics["tail_losers"] == 2
    # Of the 16 safe-loser pairs, only the loser scored 3 against the company
    # scored 2 is in the wrong order.
    assert metrics["tail_loss_auc"] == pytest.approx(15 / 16)
    # The riskiest fifth (scored 1 and 2) holds one of the two tail losers.
    assert metrics["riskiest_fifth_tail_share"] == 0.5
    assert metrics["safest_fifth_tail_share"] == 0.0
    # Eight neighbours swap places by one rank: 1 - 6 * 8 / (10 * 99).
    assert metrics["rank_correlation"] == pytest.approx(1 - 48 / 990)
    # Safest fifth (15, 30) against riskiest (-50, -10), in percentage points.
    assert metrics["top_minus_bottom_return"] == pytest.approx(22.5 - (-30.0))


def test_the_riskiest_and_safest_fifths_hold_as_many_companies() -> None:
    # Eleven companies: a fifth is 2.2, so two at each end. The two lowest
    # returns are the riskiest pair, the next two the safest.
    scored = pd.DataFrame(
        {
            "fiscal_year": 2017,
            "score": range(11),
            "next_year_return": [-9.0, -8.0, 1, 2, 3, 4, 5, 6, 7, -7.0, -6.0],
            "tail_loss": [True, True] + [False] * 7 + [True, True],
        }
    )

    metrics = ranking_metrics(scored)

    assert metrics["riskiest_fifth_tail_share"] == 0.5
    assert metrics["safest_fifth_tail_share"] == 0.5
    assert metrics["top_minus_bottom_return"] == pytest.approx(-6.5 - -8.5)


def test_pooled_years_are_ranked_within_each_year() -> None:
    # 2018's scores are all above 2017's; ranked within each year, the two
    # years still look the same, and the pooled result equals each one's.
    pooled = pd.concat([one_year(2017), one_year(2018, shift=100.0)], ignore_index=True)

    metrics = ranking_metrics(pooled)

    assert metrics["companies"] == 20
    assert metrics["tail_loss_auc"] == pytest.approx(15 / 16)
    assert metrics["riskiest_fifth_tail_share"] == 0.5
    assert metrics["rank_correlation"] == pytest.approx(1 - 48 / 990)


# Return on equity alone, kept; the five other ratios dropped.
ROE_ONLY = pd.DataFrame(
    {
        "criterion": ["roe", "net_margin", "pe"],
        "group": ["profitability", "profitability", "valuation"],
        "direction": ["keep", "drop", "drop"],
    }
)


def universe() -> pd.DataFrame:
    """Two test years of the ten companies, ranked by return on equity alone,
    bigger the safer, and with last year's price move pointing the wrong way.
    The ML risk score puts the two tail losers at the bottom."""
    years = pd.concat([one_year(2017), one_year(2018)], ignore_index=True)
    return years.assign(
        cik=range(len(years)),
        roe_score=years["score"] / 10,
        ml_risk_score=(~years["tail_loss"]).astype(float),
        market_cap=years["score"] * 1e9,
        last_year_return=-years["score"],
    ).drop(columns="score")


def test_the_ablation_table_has_a_ratios_only_row_per_profile_method_and_test_year() -> None:
    table = ablation_table(universe(), ROE_ONLY)

    ratios = table[table["layer"] == "ratios only"]
    assert len(ratios) == 3 * 2 * 3
    assert set(ratios["profile"]) == {"balanced", "risk-averse", "value-minded"}
    assert set(ratios["method"]) == {"SAW", "TOPSIS"}
    assert set(ratios["fiscal_year"]) == {"2017", "2018", "pooled"}
    # One criterion: every profile and method ranks as it does.
    assert ratios["tail_loss_auc"].tolist() == pytest.approx([15 / 16] * len(ratios))


def test_the_ml_row_adds_the_ml_risk_score_to_the_ratios_and_is_labelled_exploratory() -> None:
    table = ablation_table(universe(), ROE_ONLY)

    ml = table[table["layer"] == "+ ML"]
    assert len(ml) == 3 * 2 * 3
    # The ML score puts both tail losers below everyone else, and return on
    # equity cannot lift them out under any profile (worked by hand).
    assert ml["tail_loss_auc"].tolist() == pytest.approx([1.0] * len(ml))
    assert ml["exploratory"].all()
    assert not table.loc[table["layer"] != "+ ML", "exploratory"].any()


def test_the_baselines_are_ranked_beside_the_layers_without_a_profile() -> None:
    table = ablation_table(universe(), ROE_ONLY).set_index(["layer", "fiscal_year"])

    assert table.loc[("size baseline", "pooled"), "tail_loss_auc"] == pytest.approx(15 / 16)
    assert table.loc[("last year's price move baseline", "2017"), "tail_loss_auc"] == (
        pytest.approx(1 / 16)
    )
    assert pd.isna(table.loc[("size baseline", "2017"), "profile"])


def test_every_row_of_the_ablation_is_measured_on_the_same_companies() -> None:
    rows = universe()
    # No prior-year row: the price move is not known, so the baseline puts
    # the company in the middle of its year, as a ratio "not reported" is.
    rows.loc[0, "last_year_return"] = None
    rows.loc[1, "market_cap"] = None

    table = ablation_table(rows, ROE_ONLY)

    pooled = table[table["fiscal_year"] == "pooled"]
    assert set(pooled["companies"]) == {20}


def test_the_universe_is_test_year_rows_with_10k_text_an_audit_opinion_and_enough_ratios() -> None:
    keys = pd.DataFrame(
        {
            "cik": [1, 2, 3, 4, 1, 5],
            "fiscal_year": [2017, 2017, 2018, 2018, 2016, 2018],
            "ticker": ["A", "B", "C", "D", "A", "E"],
        }
    )
    joined = keys.assign(next_year_return=[5.0, 1.0, -9.0, 2.0, 4.0, 3.0], tail_loss=False)
    criteria = keys.assign(
        sector="Technology",
        roe_score=[0.9, 0.1, 0.5, 0.4, 0.3, 0.7],
        too_few_ratios=[False, False, True, False, False, False],
    )
    filings = keys.assign(
        in_corpus=[True, False, True, True, True, True],
        no_report=pd.array([False, pd.NA, False, False, False, True], dtype="boolean"),
    )
    rows = baselines(joined.assign(**{"Market Cap": 1e9}))
    # The ML model scores test years only.
    ml = keys[keys["fiscal_year"] >= 2017].assign(ml_risk_score=[0.2, 0.8, 0.6, 0.4, 0.5])

    table = evaluation_universe(joined, criteria, filings, rows, ml)

    # B has no 10-K text, C too few ratios, A's fiscal 2016 is a training year,
    # and E's Items 8 and 15 hold no audit opinion text.
    assert table["ticker"].tolist() == ["A", "D"]
    assert table["roe_score"].tolist() == [0.9, 0.4]
    assert table["ml_risk_score"].tolist() == [0.2, 0.4]
    assert table["last_year_return"].tolist()[0] == 4.0
    assert {"next_year_return", "tail_loss", "market_cap"} <= set(table.columns)


def test_the_llm_row_adds_the_going_concern_flag_to_the_ratios_and_ml() -> None:
    # ML ranks everyone alike here; the flag alone lifts the order to perfect.
    rows = universe().assign(ml_risk_score=0.5)
    rows["going_concern_score"] = (~rows["tail_loss"]).astype(float)

    table = ablation_table(rows, ROE_ONLY)

    llm = table[table["layer"] == "+ LLM"]
    assert len(llm) == 3 * 2 * 3
    assert llm["tail_loss_auc"].tolist() == pytest.approx([1.0] * len(llm))
    ml = table[table["layer"] == "+ ML"]
    assert ml["tail_loss_auc"].tolist() == pytest.approx([15 / 16] * len(ml))
    # It still holds the ML layer, so it is exploratory too.
    assert llm["exploratory"].all()


def test_a_failed_extraction_is_ranked_on_its_other_criteria_not_dropped() -> None:
    rows = universe()
    rows["going_concern_score"] = 1.0
    rows.loc[0, "going_concern_score"] = None

    table = ablation_table(rows, ROE_ONLY)

    pooled = table[table["fiscal_year"] == "pooled"]
    assert set(pooled["layer"]) >= {"ratios only", "+ ML", "+ LLM"}
    assert set(pooled["companies"]) == {20}


def test_without_going_concern_results_the_ablation_has_no_llm_row() -> None:
    table = ablation_table(universe(), ROE_ONLY)

    assert "+ LLM" not in set(table["layer"])
