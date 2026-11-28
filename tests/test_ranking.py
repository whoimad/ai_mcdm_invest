"""The ranker: weighted sum (SAW) and TOPSIS, and the weight profiles."""

import pandas as pd
import pytest

from ai_mcdm_invest.mcdm.ranking import (
    criterion_weights,
    group_shares,
    saw,
    topsis,
    weights_from_groups,
)

# Three companies scored on two criteria, higher better.
MATRIX = pd.DataFrame({"a": [1.0, 0.5, 0.0], "b": [0.0, 0.5, 1.0]}, index=["x", "y", "z"])


def test_saw_is_the_weighted_mean_of_the_criteria() -> None:
    score = saw(MATRIX, {"a": 0.75, "b": 0.25})

    assert score.tolist() == pytest.approx([0.75, 0.5, 0.25])
    assert score.index.tolist() == ["x", "y", "z"]


def test_saw_weights_need_not_sum_to_one() -> None:
    assert saw(MATRIX, {"a": 3.0, "b": 1.0}).tolist() == pytest.approx([0.75, 0.5, 0.25])


def test_saw_ranks_a_company_without_a_value_on_its_remaining_criteria() -> None:
    matrix = pd.DataFrame({"a": [0.8, 0.8], "b": [0.2, None], "c": [0.4, 0.4]})

    score = saw(matrix, {"a": 0.5, "b": 0.25, "c": 0.25})

    # 0.5 * 0.8 + 0.25 * 0.2 + 0.25 * 0.4; then (0.5 * 0.8 + 0.25 * 0.4) / 0.75
    assert score.tolist() == pytest.approx([0.55, 0.5 / 0.75])


def test_topsis_is_the_closeness_to_the_ideal_company() -> None:
    # Each column divided by its length (sqrt(1 + 0.36) for both), times its
    # weight (3:1, scaled to 0.75 and 0.25; TOPSIS ignores the scale). In units
    # of 1 / sqrt(1.36): the ideal is (0.75, 0.25) and the worst (0, 0). y is
    # sqrt(0.1) from the ideal and sqrt(0.225) from the worst: 0.6 of the way.
    matrix = pd.DataFrame({"a": [1.0, 0.6, 0.0], "b": [0.0, 0.6, 1.0]}, index=["x", "y", "z"])

    score = topsis(matrix, {"a": 3.0, "b": 1.0})

    assert score.tolist() == pytest.approx([0.75, 0.6, 0.25])
    assert score.index.tolist() == ["x", "y", "z"]


@pytest.mark.parametrize("constant", [0.5, 0.0])
def test_a_constant_criterion_does_not_change_the_order(constant: float) -> None:
    matrix = pd.DataFrame({"a": [1.0, 0.6, 0.0], "b": [0.0, 0.6, 1.0], "c": constant})
    weights = {"a": 3.0, "b": 1.0, "c": 2.0}

    # SAW: (3a + b + 2c) / 6. TOPSIS: c is as far from the ideal as from the
    # worst for everyone, so it adds nothing to either distance.
    assert saw(matrix, weights).tolist() == pytest.approx(
        [(3 + 2 * constant) / 6, (2.4 + 2 * constant) / 6, (1 + 2 * constant) / 6]
    )
    assert topsis(matrix, weights).tolist() == pytest.approx([0.75, 0.6, 0.25])


def test_topsis_ties_everyone_at_one_half_when_every_criterion_is_constant() -> None:
    matrix = pd.DataFrame({"a": [0.5, 0.5], "b": [0.0, 0.0]})

    assert topsis(matrix, {"a": 1.0, "b": 1.0}).tolist() == [0.5, 0.5]


def test_topsis_measures_a_company_without_a_value_on_its_remaining_criteria() -> None:
    # z lacks b: on a alone it sits at the worst, whatever b would have said.
    matrix = pd.DataFrame({"a": [1.0, 0.6, 0.0], "b": [0.0, 0.6, None]}, index=["x", "y", "z"])

    score = topsis(matrix, {"a": 1.0, "b": 1.0})

    assert score["z"] == 0.0
    assert score["x"] < score["y"]  # x is worst on b, y is best on it


RATIOS = {
    "roe_score": "profitability",
    "net_margin_score": "profitability",
    "pe_score": "valuation",
    "debt_equity_score": "balance_sheet",
    "current_ratio_score": "balance_sheet",
    "interest_coverage_score": "balance_sheet",
}


def test_a_switched_off_layer_s_weight_is_shared_out_over_the_rest() -> None:
    # Risk-averse: profitability 10%, valuation 10%, balance sheet 20%, ML 40%,
    # text 20%. With the ratios alone, the 40% they hold becomes 100%.
    weights = criterion_weights("risk-averse", RATIOS | {"ml_risk_score": "ml_risk"}, {"ratios"})

    assert weights == pytest.approx(
        {
            "roe_score": 0.125,
            "net_margin_score": 0.125,
            "pe_score": 0.25,
            "debt_equity_score": 0.5 / 3,
            "current_ratio_score": 0.5 / 3,
            "interest_coverage_score": 0.5 / 3,
        }
    )


def test_a_group_s_weight_is_split_equally_over_its_criteria_in_use() -> None:
    # Value-minded with ratios and ML: 25 / 35 / 10 / 20 of 90%. The current
    # ratio was dropped, so the balance sheet's share goes to the other two.
    criteria = {k: v for k, v in RATIOS.items() if k != "current_ratio_score"}

    weights = criterion_weights(
        "value-minded", criteria | {"ml_risk_score": "ml_risk"}, {"ratios", "ml"}
    )

    assert weights["roe_score"] == pytest.approx(0.25 / 0.9 / 2)
    assert weights["pe_score"] == pytest.approx(0.35 / 0.9)
    assert weights["debt_equity_score"] == pytest.approx(0.10 / 0.9 / 2)
    assert weights["ml_risk_score"] == pytest.approx(0.20 / 0.9)
    assert "current_ratio_score" not in weights


def test_the_balanced_profile_weighs_every_group_alike() -> None:
    weights = criterion_weights("balanced", RATIOS, {"ratios", "ml", "text"})

    assert weights["pe_score"] == pytest.approx(1 / 3)  # no ML or text criterion yet
    assert sum(weights.values()) == pytest.approx(1.0)


def test_custom_group_weights_follow_the_profile_rule() -> None:
    # Sliders at 30 / 0 / 30 / 40 / 0 (any scale): with ratios and ML on, the
    # balance sheet's 30 of 100 is split over its three ratios.
    groups = {"profitability": 30, "valuation": 0, "balance_sheet": 30, "ml_risk": 40, "text": 0}

    weights = weights_from_groups(groups, RATIOS | {"ml_risk_score": "ml_risk"}, {"ratios", "ml"})

    assert weights["roe_score"] == pytest.approx(0.15)
    assert weights["pe_score"] == 0.0
    assert weights["debt_equity_score"] == pytest.approx(0.1)
    assert weights["ml_risk_score"] == pytest.approx(0.4)


def test_a_profile_s_group_weights_give_the_profile_s_criterion_weights() -> None:
    criteria = RATIOS | {"ml_risk_score": "ml_risk"}
    groups = {"profitability": 25, "valuation": 35, "balance_sheet": 10, "ml_risk": 20, "text": 10}

    assert weights_from_groups(groups, criteria, {"ratios", "ml"}) == pytest.approx(
        criterion_weights("value-minded", criteria, {"ratios", "ml"})
    )


def test_weights_that_are_all_zero_on_the_layers_in_use_are_refused() -> None:
    groups = {"profitability": 0, "valuation": 0, "balance_sheet": 0, "ml_risk": 50, "text": 0}

    with pytest.raises(ValueError, match="above zero"):
        weights_from_groups(groups, RATIOS, {"ratios"})


def test_group_shares_sum_to_one_and_a_switched_off_group_holds_none() -> None:
    # Risk-averse with the ratios alone: 10 / 10 / 20 of 40%.
    criteria = RATIOS | {"ml_risk_score": "ml_risk", "going_concern_score": "text"}
    groups = {"profitability": 10, "valuation": 10, "balance_sheet": 20, "ml_risk": 40, "text": 20}

    shares = group_shares(groups, criteria, {"ratios"})

    assert shares == pytest.approx(
        {
            "profitability": 0.25,
            "valuation": 0.25,
            "balance_sheet": 0.5,
            "ml_risk": 0.0,
            "text": 0.0,
        }
    )


def test_a_reference_matrix_sets_topsis_s_ideal_and_scale() -> None:
    # Measured against the three companies of the first TOPSIS test, x and y
    # keep their scores there, even when ranked alone.
    reference = pd.DataFrame({"a": [1.0, 0.6, 0.0], "b": [0.0, 0.6, 1.0]}, index=["x", "y", "z"])

    score = topsis(reference.loc[["y", "x"]], {"a": 3.0, "b": 1.0}, reference=reference)

    assert score.to_dict() == pytest.approx({"y": 0.6, "x": 0.75})
