"""ML risk score: the frozen gradient-boosted tail-loss model, scored walk-forward."""

import numpy as np
import pandas as pd

from ai_mcdm_invest.criteria.ratios import tail_loss_auc
from ai_mcdm_invest.ml.risk_score import ml_risk_criterion, ml_risk_scores

COMPANIES = 400
NOISE = [f"noise_{i}" for i in range(6)]


def clean_panel(seed: int = 0) -> pd.DataFrame:
    """Clean-panel rows for fiscal 2014-2018: a company is a tail loser when its
    `signal` is in the year's lowest fifth; six noise columns carry nothing."""
    rng = np.random.default_rng(seed)
    years = []
    for fiscal_year in range(2014, 2019):
        signal = rng.normal(size=COMPANIES)
        tail_loss = signal <= np.quantile(signal, 0.2)
        years.append(
            pd.DataFrame(
                {
                    "ticker": [f"T{i}" for i in range(COMPANIES)],
                    "fiscal_year": fiscal_year,
                    "sector": "Technology",
                    "signal": signal,
                    **{name: rng.normal(size=COMPANIES) for name in NOISE},
                    "next_year_return": np.where(tail_loss, -40.0, 10.0),
                    "tail_loss": tail_loss,
                }
            )
        )
    return pd.concat(years, ignore_index=True)


def test_every_test_year_row_gets_a_tail_loss_probability() -> None:
    panel = clean_panel()

    risk = ml_risk_scores(panel)

    assert sorted(set(risk["fiscal_year"])) == [2017, 2018]
    assert len(risk) == 2 * COMPANIES
    assert risk["tail_loss_probability"].between(0, 1).all()
    test_years = panel[panel["fiscal_year"] >= 2017].reset_index(drop=True)
    safety = -risk["tail_loss_probability"]
    assert tail_loss_auc(safety, test_years["tail_loss"]) > 0.9


def probabilities(risk: pd.DataFrame, fiscal_year: int) -> np.ndarray:
    """One fiscal year's tail-loss probabilities, in clean-panel order."""
    return risk.loc[risk["fiscal_year"] == fiscal_year, "tail_loss_probability"].to_numpy()


def test_a_test_year_is_scored_by_a_model_trained_on_fiscal_2015_to_the_year_before() -> None:
    panel = clean_panel()
    risk = ml_risk_scores(panel)
    # Relabel fiscal 2014 and 2018 at random: neither trains the 2017 model.
    relabelled = panel.copy()
    outside = relabelled["fiscal_year"].isin([2014, 2018])
    relabelled.loc[outside, "tail_loss"] = np.random.default_rng(1).random(outside.sum()) < 0.2
    unchanged = ml_risk_scores(relabelled)
    # Relabel fiscal 2017 too: it trains the 2018 model, so 2018's scores move.
    relabelled.loc[relabelled["fiscal_year"] == 2017, "tail_loss"] = False
    changed = ml_risk_scores(relabelled)

    assert np.array_equal(probabilities(unchanged, 2017), probabilities(risk, 2017))
    assert np.array_equal(probabilities(unchanged, 2018), probabilities(risk, 2018))
    assert not np.array_equal(probabilities(changed, 2018), probabilities(risk, 2018))


def test_each_score_keeps_its_five_largest_feature_contributions() -> None:
    risk = ml_risk_scores(clean_panel())

    features = risk[[f"feature_{k}" for k in range(1, 6)]]
    contributions = risk[[f"contribution_{k}" for k in range(1, 6)]].to_numpy()
    # Five different features, largest first by size, from the seven.
    assert features.nunique(axis=1).eq(5).all()
    assert set(features.to_numpy().ravel()) <= {"signal", *NOISE}
    assert (np.diff(np.abs(contributions), axis=1) <= 0).all()
    # Only `signal` decides a tail loss, so it leads nearly every company's list,
    # pushing the probability up for a low signal and down for a high one.
    assert (features["feature_1"] == "signal").mean() > 0.95
    led_by_signal = risk[features["feature_1"] == "signal"]
    assert (
        (led_by_signal["contribution_1"] > 0) == (led_by_signal["tail_loss_probability"] > 0.2)
    ).mean() > 0.95


def test_the_criterion_is_each_filer_s_safety_percentile_within_its_year() -> None:
    risk = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "U", "D", "E"],
            "fiscal_year": [2017, 2017, 2017, 2017, 2018, 2018],
            "tail_loss_probability": [0.6, 0.1, 0.3, 0.9, 0.5, 0.2],
        }
    )
    # U has no SEC filer; the joined panel keeps one row per filer and year.
    joined = pd.DataFrame(
        {
            "cik": [1, 2, 3, 4, 5],
            "fiscal_year": [2017, 2017, 2017, 2018, 2018],
            "ticker": ["A", "B", "C", "D", "E"],
        }
    )

    criterion = ml_risk_criterion(risk, joined).set_index(["cik", "fiscal_year"])

    # Lowest probability safest; ranked among filers in the same year only.
    assert criterion["ml_risk_score"].to_dict() == {
        (1, 2017): 1 / 3,
        (2, 2017): 1.0,
        (3, 2017): 2 / 3,
        (4, 2018): 0.5,
        (5, 2018): 1.0,
    }
    assert criterion.loc[(1, 2017), "tail_loss_probability"] == 0.6
