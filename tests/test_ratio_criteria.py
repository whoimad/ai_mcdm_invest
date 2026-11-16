"""Ratio criteria: six textbook ratios scored as within-year, market-wide percentiles."""

import pandas as pd
import pytest

from ai_mcdm_invest.criteria.ratios import direction, learn_directions, score_ratios

# A healthy company's six ratios and the fundamentals the odd-value rules read.
HEALTHY = {
    "returnOnEquity": 0.10,
    "netProfitMargin": 0.08,
    "priceEarningsRatio": 15.0,
    "debtEquityRatio": 1.0,
    "currentRatio": 1.5,
    "interestCoverage": 5.0,
    "Net Income": 10.0,
    "Total shareholders equity": 100.0,
    "Interest Expense": 2.0,
}
SHORT_NAMES = ["roe", "net_margin", "pe", "debt_equity", "current_ratio", "interest_coverage"]
KEEP_ALL = pd.DataFrame({"criterion": SHORT_NAMES, "direction": "keep"})


def panel(*rows: dict) -> pd.DataFrame:
    """Joined-panel rows: healthy fiscal 2017 companies, with each dict's overrides."""
    return pd.DataFrame(
        [
            {
                "cik": i,
                "ticker": f"C{i}",
                "fiscal_year": 2017,
                "sector": "Technology",
                "tail_loss": False,
                **HEALTHY,
                **row,
            }
            for i, row in enumerate(rows)
        ]
    )


def scores(table: pd.DataFrame, criterion: str) -> list[float]:
    return table[f"{criterion}_score"].tolist()


@pytest.mark.parametrize(
    ("textbook_auc", "flipped_auc", "expected"),
    [
        (0.80, 0.20, "keep"),
        (0.55, 0.45, "keep"),
        (0.5499, 0.4501, "drop"),
        (0.45, 0.55, "flip"),
        (0.20, 0.80, "flip"),
        # Negative equity scored worst both ways: the textbook AUC is not low
        # enough to say "flip", but the flipped one is high.
        (0.454, 0.592, "flip"),
        # The textbook way is poor, and so is the flipped way.
        (0.40, 0.5499, "drop"),
        # Both ways pass (the signal comes from odd values): the better wins.
        (0.676, 0.661, "keep"),
        (0.60, 0.60, "keep"),
    ],
)
def test_direction_takes_the_better_way_if_it_reaches_055_and_drops_otherwise(
    textbook_auc: float, flipped_auc: float, expected: str
) -> None:
    assert direction(textbook_auc, flipped_auc) == expected


def test_scores_are_within_year_percentiles_with_higher_better() -> None:
    joined = panel(
        {"returnOnEquity": 0.30, "debtEquityRatio": 2.0},
        {"returnOnEquity": 0.10, "debtEquityRatio": 0.5},
        {"returnOnEquity": 0.20, "debtEquityRatio": 1.0},
        {"returnOnEquity": 0.20, "debtEquityRatio": 1.0},
        # Another year is ranked on its own: alone, it is the best of its year.
        {"fiscal_year": 2018, "returnOnEquity": -0.5, "debtEquityRatio": 9.0},
    )

    table = score_ratios(joined, KEEP_ALL)

    assert scores(table, "roe") == [1.0, 0.25, 0.625, 0.625, 1.0]
    assert scores(table, "debt_equity") == [0.25, 1.0, 0.625, 0.625, 1.0]  # less debt is better


def test_negative_earnings_give_the_worst_price_to_earnings() -> None:
    # The source writes a P/E of 0 when earnings are negative, which a "lower
    # is better" ranking would score best; a few rows carry a positive P/E
    # despite a loss.
    joined = panel(
        {"priceEarningsRatio": 40.0},
        {"priceEarningsRatio": 10.0},
        {"priceEarningsRatio": 0.0, "Net Income": -5.0},
        {"priceEarningsRatio": 8.0, "Net Income": -5.0},
    )

    table = score_ratios(joined, KEEP_ALL)

    assert scores(table, "pe") == [0.75, 1.0, 0.375, 0.375]
    assert table["pe_status"].tolist() == [
        "reported",
        "reported",
        "negative earnings",
        "negative earnings",
    ]


def test_negative_equity_gives_the_worst_return_on_equity_and_debt_to_equity() -> None:
    # Loss over negative equity makes a positive ROE; debt over it a negative D/E.
    joined = panel(
        {"returnOnEquity": 0.05, "debtEquityRatio": 3.0},
        {"returnOnEquity": 0.50, "debtEquityRatio": -4.0, "Total shareholders equity": -20.0},
    )

    table = score_ratios(joined, KEEP_ALL)

    assert scores(table, "roe") == [1.0, 0.5]
    assert scores(table, "debt_equity") == [1.0, 0.5]
    assert table["roe_status"].tolist() == ["reported", "negative equity"]
    assert table["debt_equity_status"].tolist() == ["reported", "negative equity"]


def test_a_missing_ratio_scores_one_half_and_is_labelled_not_reported() -> None:
    joined = panel(
        {"currentRatio": 0.5},
        {"currentRatio": 3.0},
        {"currentRatio": None},
    )

    table = score_ratios(joined, KEEP_ALL)

    assert scores(table, "current_ratio") == [0.5, 1.0, 0.5]
    assert table["current_ratio_status"].tolist() == ["reported", "reported", "not reported"]


def test_no_interest_expense_gives_the_best_interest_coverage() -> None:
    # With nothing to cover, the source writes a coverage of 0.
    joined = panel(
        {"interestCoverage": 12.0},
        {"interestCoverage": 0.0, "Interest Expense": 0.0},
        {"interestCoverage": 2.0},
    )

    table = score_ratios(joined, KEEP_ALL)

    assert scores(table, "interest_coverage") == [2 / 3, 1.0, 1 / 3]
    assert table["interest_coverage_status"].tolist() == [
        "reported",
        "no interest expense",
        "reported",
    ]


def test_a_company_missing_more_than_half_of_the_ratios_is_dropped_from_the_evaluation() -> None:
    three_missing = dict.fromkeys(["returnOnEquity", "netProfitMargin", "currentRatio"])
    joined = panel(
        {},
        three_missing,
        {**three_missing, "interestCoverage": None},
        # Undefined for a bad reason is not missing: it is scored worst.
        {**three_missing, "priceEarningsRatio": 0.0, "Net Income": -1.0},
    )

    table = score_ratios(joined, KEEP_ALL)

    assert table["ratios_not_reported"].tolist() == [0, 3, 4, 3]
    assert table["too_few_ratios"].tolist() == [False, False, True, False]


def learned(joined: pd.DataFrame) -> dict[str, tuple[float, str]]:
    directions = learn_directions(joined)
    return dict(
        zip(
            directions["criterion"],
            zip(directions["training_auc"], directions["direction"], strict=True),
            strict=True,
        )
    )


def test_directions_are_learned_on_training_years_only() -> None:
    joined = panel(
        # Fiscal 2016: the two tail losers have the lowest return on equity (as
        # a textbook screen expects), the least debt (the wrong way round), and
        # current ratios that do not separate them from the others.
        {"fiscal_year": 2016, "tail_loss": True, "returnOnEquity": 0.01,
         "debtEquityRatio": 0.2, "currentRatio": 1.0},
        {"fiscal_year": 2016, "tail_loss": True, "returnOnEquity": 0.02,
         "debtEquityRatio": 0.3, "currentRatio": 3.0},
        {"fiscal_year": 2016, "returnOnEquity": 0.20, "debtEquityRatio": 2.0,
         "currentRatio": 2.0},
        {"fiscal_year": 2016, "returnOnEquity": 0.30, "debtEquityRatio": 3.0,
         "currentRatio": 2.5},
        # A test year pointing the other way is not looked at.
        {"fiscal_year": 2017, "tail_loss": True, "returnOnEquity": 0.90},
        {"fiscal_year": 2017, "returnOnEquity": -0.50},
    )  # fmt: skip

    directions = learned(joined)

    assert directions["roe"] == (1.0, "keep")
    assert directions["debt_equity"] == (0.0, "flip")
    assert directions["current_ratio"] == (0.5, "drop")
    assert directions["net_margin"] == (0.5, "drop")  # the same for everyone


def test_leverage_flips_when_negative_equity_is_risky_but_more_debt_is_safer() -> None:
    # The leverage pattern in the data: negative equity is the riskiest, but
    # among the others the least indebted fare worst. Taken the textbook way,
    # the two cancel out; flipped with negative equity still worst, they agree.
    joined = panel(
        {"fiscal_year": 2016, "tail_loss": True, "debtEquityRatio": -2.0,
         "Total shareholders equity": -10.0},
        {"fiscal_year": 2016, "tail_loss": True, "debtEquityRatio": 0.1},
        {"fiscal_year": 2016, "debtEquityRatio": 0.2},
        {"fiscal_year": 2016, "debtEquityRatio": 2.0},
        {"fiscal_year": 2016, "debtEquityRatio": 3.0},
    )  # fmt: skip

    directions = learn_directions(joined).set_index("criterion")

    assert directions.loc["debt_equity", "training_auc"] == 0.5
    assert directions.loc["debt_equity", "flipped_training_auc"] == 1.0
    assert directions.loc["debt_equity", "direction"] == "flip"


def test_a_missing_ratio_does_not_count_towards_the_training_auc() -> None:
    joined = panel(
        {"fiscal_year": 2015, "tail_loss": True, "currentRatio": 0.5},
        {"fiscal_year": 2015, "tail_loss": True, "currentRatio": 0.8},
        {"fiscal_year": 2015, "currentRatio": 2.0},
        {"fiscal_year": 2015, "currentRatio": 3.0},
        {"fiscal_year": 2015, "currentRatio": None},
        {"fiscal_year": 2015, "currentRatio": None},
    )

    assert learned(joined)["current_ratio"] == (1.0, "keep")


def test_a_ratio_undefined_for_a_bad_reason_counts_as_worst_in_the_training_auc() -> None:
    joined = panel(
        {"fiscal_year": 2014, "tail_loss": True, "priceEarningsRatio": 0.0, "Net Income": -3.0},
        {"fiscal_year": 2014, "priceEarningsRatio": 30.0},
        {"fiscal_year": 2014, "priceEarningsRatio": 60.0},
    )

    assert learned(joined)["pe"] == (1.0, "keep")


def test_a_flipped_criterion_scores_the_other_way_but_negative_equity_stays_worst() -> None:
    joined = panel(
        {"debtEquityRatio": 0.5},
        {"debtEquityRatio": 2.0},
        {"debtEquityRatio": -1.0, "Total shareholders equity": -10.0},
    )
    directions = KEEP_ALL.assign(
        direction=KEEP_ALL["criterion"].map({"debt_equity": "flip"}).fillna("keep")
    )

    table = score_ratios(joined, directions)

    assert scores(table, "debt_equity") == [2 / 3, 1.0, 1 / 3]  # more debt scores higher


def test_a_dropped_criterion_is_not_scored_but_its_status_is_kept() -> None:
    joined = panel({}, {"currentRatio": None})
    directions = KEEP_ALL.assign(
        direction=KEEP_ALL["criterion"].map({"current_ratio": "drop"}).fillna("keep")
    )

    table = score_ratios(joined, directions)

    assert "current_ratio_score" not in table.columns
    assert table["current_ratio_status"].tolist() == ["reported", "not reported"]


def test_the_sector_percentile_is_kept_beside_the_score_as_context() -> None:
    joined = panel(
        {"returnOnEquity": 0.10, "sector": "Utilities"},
        {"returnOnEquity": 0.05, "sector": "Utilities"},
        {"returnOnEquity": 0.40},
        {"returnOnEquity": None, "sector": "Utilities"},
    )

    table = score_ratios(joined, KEEP_ALL)

    assert scores(table, "roe") == [2 / 3, 1 / 3, 1.0, 0.5]
    assert table["roe_sector_score"].tolist()[:3] == [1.0, 0.5, 1.0]
    assert pd.isna(table["roe_sector_score"].iloc[3])  # not reported, and not scored


def test_only_a_coverage_of_zero_without_interest_expense_scores_best() -> None:
    joined = panel(
        {"interestCoverage": 3.0},
        {"interestCoverage": 1.0, "Interest Expense": 0.0},
        {"interestCoverage": None, "Interest Expense": 0.0},
    )

    table = score_ratios(joined, KEEP_ALL)

    assert table["interest_coverage_status"].tolist() == ["reported", "reported", "not reported"]
