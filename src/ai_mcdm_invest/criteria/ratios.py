"""Ratio criteria stage: six textbook ratios, scored so that they can be combined."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ai_mcdm_invest.data.panel import TEST_YEARS

# Direction rule (decision 4.3, revised 16 November): a criterion is used the
# way, textbook or flipped, with the higher training tail-loss AUC, if that
# AUC reaches this value; otherwise it is dropped.
MIN_AUC = 0.55

Direction = Literal["keep", "flip", "drop"]


@dataclass(frozen=True)
class Criterion:
    """One textbook ratio: its short name, panel column, group, and whether a
    textbook screen prefers it high."""

    name: str
    column: str
    group: str
    higher_is_better: bool


CRITERIA = [
    Criterion("roe", "returnOnEquity", "profitability", True),
    Criterion("net_margin", "netProfitMargin", "profitability", True),
    Criterion("pe", "priceEarningsRatio", "valuation", False),
    Criterion("debt_equity", "debtEquityRatio", "balance_sheet", False),
    Criterion("current_ratio", "currentRatio", "balance_sheet", True),
    Criterion("interest_coverage", "interestCoverage", "balance_sheet", True),
]
ID_COLUMNS = ["cik", "ticker", "fiscal_year", "sector"]
NET_INCOME = "Net Income"
EQUITY = "Total shareholders equity"
INTEREST_EXPENSE = "Interest Expense"
# A ratio that is simply missing neither helps nor hurts.
NOT_REPORTED_SCORE = 0.5
# A company missing more than this many of the six ratios leaves the evaluation.
MAX_NOT_REPORTED = len(CRITERIA) // 2

# A ratio's status: reported as a number, undefined for a bad reason (scored
# worst) or a good one (scored best), or missing (scored neutral).
REPORTED = "reported"
NEGATIVE_EARNINGS = "negative earnings"
NEGATIVE_EQUITY = "negative equity"
NO_INTEREST_EXPENSE = "no interest expense"
NOT_REPORTED = "not reported"
WORST = {NEGATIVE_EARNINGS, NEGATIVE_EQUITY}
BEST = {NO_INTEREST_EXPENSE}


def direction(textbook_auc: float, flipped_auc: float) -> Direction:
    """Keep a criterion that separates tail losers the textbook way, flip one that
    does so better the other way, drop one that does neither.

    The two AUCs do not add up to one: odd values stay worst or best both ways.
    A tie goes to the textbook way.
    """
    if max(textbook_auc, flipped_auc) < MIN_AUC:
        return "drop"
    return "keep" if textbook_auc >= flipped_auc else "flip"


def learn_directions(joined: pd.DataFrame) -> pd.DataFrame:
    """Each criterion's direction, learned on the training years (those before the
    test years) from its tail-loss AUC taken both ways.

    Each AUC is that of the score, textbook or flipped, with odd values worst or
    best either way, among rows that do not lack the ratio: the chance that a
    company avoiding a tail loss scores higher than one suffering it (ties count
    half).
    """
    training = joined[joined["fiscal_year"] < TEST_YEARS.start]
    rows = []
    for criterion in CRITERIA:
        status = ratio_status(training, criterion)
        counted = status != NOT_REPORTED
        textbook_auc, flipped_auc = (
            tail_loss_auc(
                _percentile(training, criterion, status, flip=flip)[counted],
                training.loc[counted, "tail_loss"],
            )
            for flip in (False, True)
        )
        rows.append(
            {
                "criterion": criterion.name,
                "column": criterion.column,
                "group": criterion.group,
                "higher_is_better": criterion.higher_is_better,
                "training_auc": textbook_auc,
                "flipped_training_auc": flipped_auc,
                "direction": direction(textbook_auc, flipped_auc),
            }
        )
    return pd.DataFrame(rows)


def scored_criteria(directions: pd.DataFrame) -> dict[str, str]:
    """Each scored criterion's score column and its group: the kept and flipped ones."""
    used = directions[directions["direction"] != "drop"]
    return {
        f"{name}_score": group for name, group in zip(used["criterion"], used["group"], strict=True)
    }


def tail_loss_auc(score: pd.Series, tail_loss: pd.Series) -> float:
    """The chance that a company avoiding a tail loss scores higher than a tail
    loser, ties counting half (the Mann-Whitney form of the AUC)."""
    safe = ~tail_loss.astype(bool)
    n_safe, n_losers = int(safe.sum()), int((~safe).sum())
    ranks = score.rank()
    return float((ranks[safe].sum() - n_safe * (n_safe + 1) / 2) / (n_safe * n_losers))


def score_ratios(joined: pd.DataFrame, directions: pd.DataFrame) -> pd.DataFrame:
    """Score each kept or flipped criterion as a within-year, market-wide percentile,
    oriented so that higher is better; a dropped criterion gets no score.

    Every criterion keeps its status, and a company with more than half of the six
    ratios not reported is marked `too_few_ratios`, to leave the evaluation.
    """
    learned = directions.set_index("criterion")["direction"]
    table = joined[ID_COLUMNS].copy()
    for criterion in CRITERIA:
        status = ratio_status(joined, criterion)
        if learned[criterion.name] != "drop":
            flip = learned[criterion.name] == "flip"
            percentile = _percentile(joined, criterion, status, flip=flip)
            table[f"{criterion.name}_score"] = percentile.fillna(NOT_REPORTED_SCORE)
            table[f"{criterion.name}_sector_score"] = _percentile(
                joined, criterion, status, flip=flip, by=["fiscal_year", "sector"]
            )
        table[f"{criterion.name}_status"] = status
    not_reported = table[[f"{c.name}_status" for c in CRITERIA]] == NOT_REPORTED
    table["ratios_not_reported"] = not_reported.sum(axis=1)
    table["too_few_ratios"] = table["ratios_not_reported"] > MAX_NOT_REPORTED
    return table


def _percentile(
    joined: pd.DataFrame,
    criterion: Criterion,
    status: pd.Series,
    *,
    flip: bool,
    by: list[str] | None = None,
) -> pd.Series:
    """Percentile of the ratio within each year (or each group in `by`), higher
    better, in the textbook direction or flipped; worst or best for odd values
    whatever the direction, empty where the ratio is not reported."""
    sign = 1 if criterion.higher_is_better != flip else -1
    oriented = (sign * joined[criterion.column]).where(status == REPORTED)
    oriented[status.isin(WORST)] = -float("inf")
    oriented[status.isin(BEST)] = float("inf")
    return oriented.groupby([joined[column] for column in by or ["fiscal_year"]]).rank(pct=True)


def ratio_status(joined: pd.DataFrame, criterion: Criterion) -> pd.Series:
    """Whether each row's ratio is reported, undefined for a bad reason, or missing.

    Price-to-earnings is undefined when net income is negative; the source then
    writes 0, so a ratio at or below 0 counts as negative earnings too.
    Return on equity and debt-to-equity are undefined when equity is negative.
    Interest coverage is undefined when there is no interest to cover; the
    source then writes 0, but the coverage is unbounded, so it scores best.
    """
    value = joined[criterion.column]
    status = pd.Series(REPORTED, index=joined.index).where(value.notna(), NOT_REPORTED)
    if criterion.name == "pe":
        status[(joined[NET_INCOME] < 0) | (value <= 0)] = NEGATIVE_EARNINGS
    if criterion.name == "interest_coverage":
        status[(joined[INTEREST_EXPENSE] == 0) & (value == 0)] = NO_INTEREST_EXPENSE
    if criterion.name in {"roe", "debt_equity"}:
        status[joined[EQUITY] < 0] = NEGATIVE_EQUITY
    return status
