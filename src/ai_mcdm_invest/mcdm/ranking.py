"""The ranker: weighted sum (SAW), with TOPSIS as a robustness check.

Both take a matrix of criteria (one row per company, one column per criterion,
every criterion oriented so that higher is better) and a weight per column.
"""

from collections.abc import Collection, Mapping

import pandas as pd

# Weight profiles by criterion group (decision 4.10), fixed before testing and
# never fitted to returns. Each group's weight is split equally over its
# criteria in use.
PROFILES = {
    "balanced": {
        "profitability": 0.20,
        "valuation": 0.20,
        "balance_sheet": 0.20,
        "ml_risk": 0.20,
        "text": 0.20,
    },
    "risk-averse": {
        "profitability": 0.10,
        "valuation": 0.10,
        "balance_sheet": 0.20,
        "ml_risk": 0.40,
        "text": 0.20,
    },
    "value-minded": {
        "profitability": 0.25,
        "valuation": 0.35,
        "balance_sheet": 0.10,
        "ml_risk": 0.20,
        "text": 0.10,
    },
}
# The layer each criterion group belongs to; the ablation switches layers on in turn.
GROUP_LAYER = {
    "profitability": "ratios",
    "valuation": "ratios",
    "balance_sheet": "ratios",
    "ml_risk": "ml",
    "text": "text",
}


def criterion_weights(
    profile: str, criteria: Mapping[str, str], layers: Collection[str]
) -> dict[str, float]:
    """Each criterion's weight under a profile, with only `layers` switched on.

    `criteria` maps each criterion in use to its group. A group's weight is
    split equally over its criteria; a group with none (its layer switched off,
    not built yet, or every criterion dropped) loses its weight, and the rest
    are rescaled to sum to one.
    """
    on = {name: group for name, group in criteria.items() if GROUP_LAYER[group] in layers}
    per_group = pd.Series(on).value_counts()
    weights = {name: PROFILES[profile][group] / per_group[group] for name, group in on.items()}
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def saw(matrix: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Weighted sum: each company's weighted mean of its criteria.

    Weights need not sum to one. A company without a value for a criterion is
    scored on its remaining criteria, their weights rescaled.
    """
    values = matrix[list(weights)]
    weight = pd.Series(weights, dtype=float)
    present_weight = values.notna().mul(weight).sum(axis=1)
    return values.mul(weight).sum(axis=1) / present_weight


def topsis(matrix: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """TOPSIS: each company's closeness to the ideal company, from 0 to 1.

    Each column is divided by its length and multiplied by its weight; the
    ideal company has every column's best value, the worst company every
    column's worst. The closeness is the distance to the worst divided by the
    sum of the distances to the ideal and to the worst. Weights need not sum to
    one: scaling them all scales both distances alike.

    A constant column is as far from the ideal as from the worst for everyone,
    so it adds nothing (an all-zero column, of length 0, is left out). When
    every column is constant, everyone ties at 0.5. A company without a value
    for a criterion is measured on its remaining criteria.
    """
    values = matrix[list(weights)]
    weight = pd.Series(weights, dtype=float) / sum(weights.values())
    length = ((values**2).sum() ** 0.5).replace(0.0, float("nan"))
    weighted = values.div(length).mul(weight)
    to_ideal = ((weighted - weighted.max()) ** 2).sum(axis=1) ** 0.5
    to_worst = ((weighted - weighted.min()) ** 2).sum(axis=1) ** 0.5
    total = to_ideal + to_worst
    return (to_worst / total).where(total > 0, 0.5)
