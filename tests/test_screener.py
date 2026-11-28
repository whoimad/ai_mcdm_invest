"""The screener's logic: scores, filters and the summary of the rows shown."""

import pandas as pd
import pytest

from ai_mcdm_invest.mcdm.ranking import topsis
from ai_mcdm_invest.screener import filter_rows, score, study_layer_row, tail_shares

# Five companies on two criteria; the last is outside the study's universe.
TABLE = pd.DataFrame(
    {
        "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
        "filer_name": ["Alpha Corp", "Beta Inc", "Gamma Co", "Delta Ltd", "Echo plc"],
        "sector": ["Tech", "Tech", "Energy", "Energy", "Utilities"],
        "status": ["clear", "flagged", "clear", "not in corpus", "clear"],
        "a": [1.0, 0.6, 0.0, 0.3, 0.9],
        "b": [0.0, 0.6, 1.0, 0.2, 0.9],
        "in_study": [True, True, True, True, False],
    }
)
WEIGHTS = {"a": 3.0, "b": 1.0}


def test_saw_scores_each_company_on_its_own_criteria() -> None:
    assert score(TABLE, WEIGHTS, "SAW").tolist() == pytest.approx([0.75, 0.6, 0.25, 0.275, 0.9])


def test_topsis_measures_every_company_against_the_study_s_universe() -> None:
    scores = score(TABLE, WEIGHTS, "TOPSIS")

    # The universe's companies keep the score the study gives them, and the
    # company outside it is measured on the same scale.
    in_study = TABLE[TABLE["in_study"]]
    assert scores[TABLE["in_study"]].tolist() == pytest.approx(topsis(in_study, WEIGHTS).tolist())
    assert scores[4] > scores[:4].max()
    assert scores[4] <= 1.0


def test_filters_hide_rows_and_keep_their_scores() -> None:
    scored = TABLE.assign(score=score(TABLE, WEIGHTS, "TOPSIS"))

    shown = filter_rows(scored, sectors=["Tech", "Energy"], statuses=["clear"], search="")

    assert shown["ticker"].tolist() == ["AAA", "CCC"]
    assert shown["score"].tolist() == scored.loc[[0, 2], "score"].tolist()


def test_no_filter_set_shows_every_row() -> None:
    assert len(filter_rows(TABLE, sectors=[], statuses=[], search="  ")) == 5


@pytest.mark.parametrize("search", ["bbb", "beta", " Beta "])
def test_search_matches_the_ticker_or_the_name_in_any_case(search: str) -> None:
    assert filter_rows(TABLE, sectors=[], statuses=[], search=search)["ticker"].tolist() == ["BBB"]


def test_tail_shares_are_the_share_of_tail_losers_in_each_fifth() -> None:
    # Ten companies, two tail losers: one among the two lowest scores, one in
    # the middle, none among the two highest.
    rows = pd.DataFrame(
        {
            "fiscal_year": 2017,
            "score": [0.1 * k for k in range(10)],
            "next_year_return": [-80.0, 5, 3, 2, -70, 1, 4, 6, 8, 9],
            "tail_loss": [True, False, False, False, True, False, False, False, False, False],
        }
    )

    assert tail_shares(rows) == pytest.approx({"riskiest": 0.5, "safest": 0.0})


def test_tail_shares_are_empty_without_a_tail_loser() -> None:
    rows = pd.DataFrame(
        {"fiscal_year": 2017, "score": [0.1, 0.2], "next_year_return": 1.0, "tail_loss": False}
    )

    assert tail_shares(rows) is None


def test_the_layers_in_use_are_named_after_the_study_s_ablation_row() -> None:
    assert study_layer_row({"ratios"}) == "ratios only"
    assert study_layer_row({"ratios", "ml", "text"}) == "+ LLM"
    assert study_layer_row({"ml", "text"}) is None
