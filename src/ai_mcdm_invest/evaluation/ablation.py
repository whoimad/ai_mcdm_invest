"""Evaluation stage: the baselines and the layer ablation table.

The ablation ranks the same companies with one more layer of criteria switched
on at each row, under each weight profile, and describes how well each ranking
separates the next year's tail losers.
"""

import pandas as pd

from ai_mcdm_invest.criteria.ratios import NOT_REPORTED_SCORE, scored_criteria, tail_loss_auc
from ai_mcdm_invest.data.panel import TEST_YEARS
from ai_mcdm_invest.mcdm.ranking import PROFILES, criterion_weights, saw, topsis
from ai_mcdm_invest.ml.risk_score import ML_GROUP, ML_SCORE_COLUMN

MARKET_CAP = "Market Cap"
# The riskiest and safest fifths: the lowest and highest 20% of scores in a year.
FIFTH = 0.2
# The ablation's rows: each switches on one more layer of criteria.
LAYERS = {
    "ratios only": {"ratios"},
    "+ ML": {"ratios", "ml"},
    "+ LLM": {"ratios", "ml", "text"},
}
# The text layer's one criterion (pre-registration, section 3): the LLM's
# going-concern flag, clear 1 and flagged 0, empty when the extraction failed.
TEXT_LAYER = "text"
TEXT_SCORE_COLUMN = "going_concern_score"
TEXT_GROUP = "text"
# The ML numbers are exploratory (decision 4.8): its settings were never tuned
# on the test years, but its target and model family were chosen while
# looking at them.
EXPLORATORY_LAYER = "ml"
METHODS = {"SAW": saw, "TOPSIS": topsis}
# Baselines outside MCDM (decision 4.2), each a column where higher is safer.
BASELINES = {
    "size baseline": "market_cap",
    "last year's price move baseline": "last_year_return",
}
POOLED = "pooled"
KEY = ["cik", "fiscal_year"]


def baselines(joined: pd.DataFrame) -> pd.DataFrame:
    """Size and last year's price move for each joined-panel row, kept outside MCDM.

    `market_cap` is the panel's market cap at the fiscal year end.
    `last_year_return` (in %) is the same filer's prior fiscal year's next-year
    return: the price move over the calendar year this fiscal year ends in. It
    is empty when the filer has no row for the prior fiscal year.
    """
    table = joined[[*KEY, "ticker"]].copy()
    table["market_cap"] = joined[MARKET_CAP]
    prior = joined[[*KEY, "next_year_return"]].assign(fiscal_year=joined["fiscal_year"] + 1)
    table = table.merge(prior, on=KEY, how="left", validate="one_to_one")
    return table.rename(columns={"next_year_return": "last_year_return"})


def evaluation_universe(
    joined: pd.DataFrame,
    criteria: pd.DataFrame,
    filings: pd.DataFrame,
    baseline_table: pd.DataFrame,
    ml_criterion: pd.DataFrame,
) -> pd.DataFrame:
    """The companies every layer is measured on (decision 5.2): test-year rows
    whose 10-K text is in the corpus, whose Items 8 and 15 hold audit opinion
    text (not `no_report`), and that report at least half the ratios.

    One row per filer and test year, with its ticker, ratio scores,
    `ml_risk_score`, baselines, `next_year_return` and `tail_loss`.
    """
    scores = criteria.drop(columns=["ticker", "sector"])
    table = (
        joined[[*KEY, "ticker", "next_year_return", "tail_loss"]]
        .merge(scores, on=KEY, validate="one_to_one")
        .merge(filings[[*KEY, "in_corpus", "no_report"]], on=KEY, validate="one_to_one")
        .merge(baseline_table.drop(columns="ticker"), on=KEY, validate="one_to_one")
        .merge(ml_criterion[[*KEY, ML_SCORE_COLUMN]], on=KEY, how="left", validate="one_to_one")
    )
    kept = (
        table["fiscal_year"].isin(TEST_YEARS)
        & table["in_corpus"]
        & ~table["no_report"].fillna(True).astype(bool)
        & ~table["too_few_ratios"]
    )
    return table[kept].drop(columns=["in_corpus", "no_report"]).reset_index(drop=True)


def ablation_table(universe: pd.DataFrame, directions: pd.DataFrame) -> pd.DataFrame:
    """The layer ablation: each layer under each profile and method, then each
    baseline, described by `ranking_metrics` per test year and pooled.

    `universe` is the evaluation universe: one row per company and test year,
    with `fiscal_year`, `next_year_return`, `tail_loss`, the score of each
    criterion `directions` keeps or flips, `ml_risk_score`, and the baseline
    columns. Rows that include the ML layer are marked `exploratory`. A
    baseline value that is missing scores the middle of its year, as a ratio
    not reported does, so every row is measured on the same companies.

    The "+ LLM" rows appear only when `universe` has `going_concern_score`; a
    company without one (a failed extraction) is ranked on its other criteria.
    """
    criteria = {**scored_criteria(directions), ML_SCORE_COLUMN: ML_GROUP}
    has_text = TEXT_SCORE_COLUMN in universe.columns
    if has_text:
        criteria[TEXT_SCORE_COLUMN] = TEXT_GROUP
    rankings = []
    for layer, layers in LAYERS.items():
        if TEXT_LAYER in layers and not has_text:
            continue
        exploratory = EXPLORATORY_LAYER in layers
        for profile in PROFILES:
            weights = criterion_weights(profile, criteria, layers)
            for method, rank in METHODS.items():
                score = rank(universe, weights)
                labels = {"layer": layer, "profile": profile, "method": method}
                rankings.append(({**labels, "exploratory": exploratory}, score))
    for layer, column in BASELINES.items():
        percentile = universe[column].groupby(universe["fiscal_year"]).rank(pct=True)
        score = percentile.fillna(NOT_REPORTED_SCORE)
        labels = {"layer": layer, "profile": None, "method": None, "exploratory": False}
        rankings.append((labels, score))

    year = universe["fiscal_year"]
    samples = [(str(y), year == y) for y in sorted(set(year))] + [(POOLED, year.notna())]
    rows = []
    for labels, score in rankings:
        scored = universe.assign(score=score)
        for fiscal_year, in_sample in samples:
            metrics = ranking_metrics(scored[in_sample])
            rows.append({**labels, "fiscal_year": fiscal_year, **metrics})
    return pd.DataFrame(rows)


def ranking_metrics(scored: pd.DataFrame) -> dict[str, float]:
    """How well a `score` (higher is safer) separates the next year's tail losers.

    `scored` has `fiscal_year`, `score`, `next_year_return` and `tail_loss`;
    rows without a score are left out. Scores and returns are ranked within
    each fiscal year first, so pooled years are compared like for like:

    - `tail_loss_auc`: the chance that a company avoiding a tail loss scores
      higher than a tail loser (ties count half);
    - `riskiest_fifth_tail_share`, `safest_fifth_tail_share`: the share of the
      tail losers that fall in each year's lowest or highest fifth of scores
      (a random ranking puts 20% in each); both fifths are cut alike, so they
      hold as many companies when the year's count is not a multiple of five;
    - `rank_correlation`: the correlation of the score's rank with the
      return's rank (Spearman's, within a single year);
    - `top_minus_bottom_return`: the mean next-year return of the safest fifth
      minus that of the riskiest, in percentage points.
    """
    rows = scored[scored["score"].notna()]
    year = rows["fiscal_year"]
    by_year = rows["score"].groupby(year)
    score_rank = by_year.rank(pct=True)
    return_rank = rows["next_year_return"].groupby(year).rank(pct=True)
    tail_loss = rows["tail_loss"].astype(bool)
    fifth = FIFTH * year.map(year.value_counts())
    riskiest = by_year.rank() <= fifth
    safest = by_year.rank(ascending=False) <= fifth
    returns = rows["next_year_return"]
    return {
        "companies": len(rows),
        "tail_losers": int(tail_loss.sum()),
        "tail_loss_auc": tail_loss_auc(score_rank, tail_loss),
        "riskiest_fifth_tail_share": float(tail_loss[riskiest].sum() / tail_loss.sum()),
        "safest_fifth_tail_share": float(tail_loss[safest].sum() / tail_loss.sum()),
        "rank_correlation": float(score_rank.corr(return_rank)),
        "top_minus_bottom_return": float(returns[safest].mean() - returns[riskiest].mean()),
    }
