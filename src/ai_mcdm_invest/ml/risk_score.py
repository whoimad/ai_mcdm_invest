"""ML risk score stage: the probability that a company lands in the year's worst
20% of returns next year, from gradient-boosted trees on the 221 fundamentals.

The model keeps the prototype's settings (decision 4.6) and is never tuned.
It is trained walk-forward on the full clean panel, linked to a 10-K or not.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from ai_mcdm_invest.criteria.ratios import tail_loss_auc
from ai_mcdm_invest.data.panel import NON_FUNDAMENTAL_COLUMNS, TEST_YEARS

# The prototype's first training year; fiscal 2014 is left out of training.
FIRST_TRAINING_YEAR = 2015
# Frozen as the prototype ran them (decision 4.6); never tuned.
TREES = 400
PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 10,
    "tree_method": "hist",
    "seed": 0,
}
# The criterion's score column, and its group in the weight profiles.
ML_SCORE_COLUMN = "ml_risk_score"
ML_GROUP = "ml_risk"
# How many of each company's feature contributions are kept to explain its score.
TOP_CONTRIBUTIONS = 5


def ml_risk_scores(clean: pd.DataFrame) -> pd.DataFrame:
    """Each test-year clean-panel row's `tail_loss_probability`, from a model
    trained on fiscal `FIRST_TRAINING_YEAR` to the year before.

    One row per clean-panel row in the test years, in clean-panel order, with
    `ticker` and `fiscal_year`. `feature_1` to `feature_5` name the features
    whose contributions to the row's score are largest in size, largest first,
    and `contribution_1` to `contribution_5` give them, in log-odds of a tail
    loss (positive raises the probability). They are the model's own
    contribution output (SHAP values), so no extra library is needed.
    """
    features = [c for c in clean.columns if c not in NON_FUNDAMENTAL_COLUMNS]
    years = []
    for test_year in TEST_YEARS:
        training = clean[clean["fiscal_year"].between(FIRST_TRAINING_YEAR, test_year - 1)]
        scored = clean[clean["fiscal_year"] == test_year]
        model = xgb.train(
            PARAMS,
            xgb.DMatrix(training[features], label=training["tail_loss"]),
            num_boost_round=TREES,
        )
        matrix = xgb.DMatrix(scored[features])
        table = scored[["ticker", "fiscal_year"]].assign(
            tail_loss_probability=model.predict(matrix)
        )
        # The last column of the contribution output is the bias, shared by all.
        contributions = model.predict(matrix, pred_contribs=True)[:, :-1]
        years.append(
            pd.concat([table, _top_contributions(contributions, features, table.index)], axis=1)
        )
    return pd.concat(years, ignore_index=True)


def _top_contributions(
    contributions: np.ndarray, features: list[str], index: pd.Index
) -> pd.DataFrame:
    """Each row's `TOP_CONTRIBUTIONS` largest contributions in size, largest first."""
    order = np.argsort(-np.abs(contributions), axis=1, kind="stable")[:, :TOP_CONTRIBUTIONS]
    names = np.asarray(features)[order]
    values = np.take_along_axis(contributions, order, axis=1)
    columns = {}
    for k in range(TOP_CONTRIBUTIONS):
        columns[f"feature_{k + 1}"] = names[:, k]
        columns[f"contribution_{k + 1}"] = values[:, k]
    return pd.DataFrame(columns, index=index)


def ml_risk_criterion(risk: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    """The ML risk criterion for each joined-panel row the model scored.

    `ml_risk_score` is the row's percentile within its fiscal year over the
    joined panel, as the ratio scores are, oriented so that higher is safer
    (a lower tail-loss probability). One row per filer (`cik`) and test year,
    with `ticker` and `tail_loss_probability`.
    """
    table = joined[["cik", "fiscal_year", "ticker"]].merge(
        risk[["ticker", "fiscal_year", "tail_loss_probability"]],
        on=["ticker", "fiscal_year"],
        validate="one_to_one",
    )
    safety = -table["tail_loss_probability"]
    table[ML_SCORE_COLUMN] = safety.groupby(table["fiscal_year"]).rank(pct=True)
    return table


def ml_auc_by_year(risk: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """The model's out-of-time tail-loss AUC per test year, on the full clean
    panel, with the rows scored."""
    table = risk.merge(clean[["ticker", "fiscal_year", "tail_loss"]], on=["ticker", "fiscal_year"])
    by_year = table.groupby("fiscal_year")
    auc = {
        fiscal_year: tail_loss_auc(-rows["tail_loss_probability"], rows["tail_loss"])
        for fiscal_year, rows in by_year
    }
    return pd.DataFrame({"rows": by_year.size(), "tail_loss_auc": pd.Series(auc)})
