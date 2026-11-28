"""The screener's logic: scores, filters and the summary of the rows shown."""

import pandas as pd
import pytest

from ai_mcdm_invest.mcdm.ranking import topsis
from ai_mcdm_invest.screener import (
    KEYWORD_ONLY,
    LLM_ONLY,
    card_company,
    clicked_company,
    edgar_url,
    filter_rows,
    score,
    study_layer_row,
    tail_shares,
    text_evidence,
)

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


def test_the_edgar_link_is_the_filing_s_folder_named_by_its_accession() -> None:
    assert (
        edgar_url(1750, "0001047469-17-004528")
        == "https://www.sec.gov/Archives/edgar/data/1750/000104746917004528/"
    )


SHOWN = pd.DataFrame(
    {"cik": [30, 10, 20], "fiscal_year": 2017, "ticker": ["CCC", "AAA", "BBB"]}, index=[7, 2, 5]
)


def test_the_card_shows_the_chosen_company_when_it_is_shown() -> None:
    assert card_company(SHOWN, (20, 2017))["ticker"] == "BBB"


@pytest.mark.parametrize("chosen", [None, (40, 2017), (20, 2018)])
def test_the_card_falls_back_to_the_first_row_shown(chosen: tuple[int, int] | None) -> None:
    assert card_company(SHOWN, chosen)["ticker"] == "CCC"


def test_no_card_when_no_row_is_shown() -> None:
    assert card_company(SHOWN.iloc[:0], (20, 2017)) is None


def test_a_clicked_cell_picks_the_company_of_its_row() -> None:
    # Positions count the rows shown, whatever their index.
    assert clicked_company(SHOWN, cells=[(1, "Sector")], rows=[]) == (10, 2017)


def test_a_ticked_row_picks_its_company_when_no_cell_is_clicked() -> None:
    assert clicked_company(SHOWN, cells=[], rows=[2]) == (20, 2017)


def test_nothing_clicked_picks_no_company() -> None:
    assert clicked_company(SHOWN, cells=[], rows=[]) is None


# Chunks of three filings: one both flag, one only kw-1, one only the LLM.
FACTS = {
    "auditor_opinion": True,
    "this_company": True,
    "current_year": True,
    "doubt_expressed": True,
    "alleviated": False,
}
QUOTES = pd.DataFrame(
    [
        {"cik": 1, "fiscal_year": 2017, "chunk": 1, "item": "8", "status": "flagged",
         "quote": "Second doubt.", **FACTS, "keyword_match": None},
        {"cik": 1, "fiscal_year": 2017, "chunk": 0, "item": "8", "status": "flagged",
         "quote": "First doubt.", **FACTS, "keyword_match": "Raise substantial doubt."},
        {"cik": 2, "fiscal_year": 2017, "chunk": 0, "item": "7", "status": "clear",
         "quote": "", **dict.fromkeys(FACTS, False), "keyword_match": "Doubt was alleviated."},
        {"cik": 3, "fiscal_year": 2017, "chunk": 0, "item": "8", "status": "flagged",
         "quote": "Doubt.", **FACTS, "keyword_match": None},
    ]
)  # fmt: skip


def company(cik: int, status: str, keyword_flag: object) -> pd.Series:
    return pd.Series(
        {"cik": cik, "fiscal_year": 2017, "status": status, "keyword_flag": keyword_flag}
    )


def test_both_flag_gives_the_llm_quotes_in_order_and_kw_1_s_sentence() -> None:
    evidence = text_evidence(company(1, "flagged", True), QUOTES)

    assert evidence.quotes["quote"].tolist() == ["First doubt.", "Second doubt."]
    assert evidence.quotes["auditor_opinion"].all()
    assert evidence.keyword_flag is True
    assert evidence.keyword_sentence == "Raise substantial doubt."
    assert evidence.disagreement is None


def test_kw_1_alone_is_a_disagreement_with_no_llm_quote() -> None:
    evidence = text_evidence(company(2, "clear", True), QUOTES)

    assert evidence.quotes.empty
    assert evidence.keyword_sentence == "Doubt was alleviated."
    assert evidence.disagreement == KEYWORD_ONLY


def test_the_llm_alone_is_a_disagreement_with_no_kw_1_sentence() -> None:
    evidence = text_evidence(company(3, "flagged", False), QUOTES)

    assert evidence.keyword_flag is False
    assert evidence.keyword_sentence is None
    assert evidence.disagreement == LLM_ONLY


@pytest.mark.parametrize("status", ["extraction_failed", "not in corpus"])
def test_no_disagreement_without_an_llm_verdict(status: str) -> None:
    keyword_flag = True if status == "extraction_failed" else float("nan")

    evidence = text_evidence(company(4, status, keyword_flag), QUOTES)

    assert evidence.disagreement is None
    assert evidence.keyword_flag is (True if status == "extraction_failed" else None)
