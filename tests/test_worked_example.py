"""The worked example: one company's 10-K passages, quoted beside its fundamentals."""

from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.worked_example import worked_example_passages


@pytest.fixture
def filings() -> pd.DataFrame:
    """Rows of the filings table that locate ROMO's and PAPA's records in the corpus."""
    return pd.DataFrame(
        {
            "cik": [1017, 1017, 1015],
            "fiscal_year": [2015, 2017, 2015],
            "ticker": ["ROMO", "ROMO", "PAPA"],
            "in_corpus": [True, True, False],
            "corpus_year": pd.array([2015, 2017, None], dtype="Int64"),
        }
    )


def corpus_dir(fixture_raw_dir: Path) -> Path:
    return Settings(raw_dir=fixture_raw_dir).raw_corpus_dir


def test_quotes_the_going_concern_paragraphs_and_the_legal_proceedings(
    filings: pd.DataFrame, fixture_raw_dir: Path
) -> None:
    passages = worked_example_passages(filings, corpus_dir(fixture_raw_dir), "ROMO", 2015)

    assert list(passages["item"]) == ["8", "1A", "3"]
    assert set(passages["ticker"]) == {"ROMO"}
    assert set(passages["cik"]) == {1017}
    assert set(passages["fiscal_year"]) == {2015}
    statements, risk, legal = passages["passage"]
    assert statements.endswith(
        "These conditions raise substantial doubt about the Company's ability "
        "to continue as a going concern."
    )
    assert risk == (
        "We have lost money for three years. We may need more capital. "
        "If we cannot raise it, we may not be able to continue as a going concern."
    )
    assert legal == (
        "The company is party to routine litigation incidental to its business, "
        "none of which is material."
    )


def test_long_passages_are_cut_at_a_sentence_end(
    filings: pd.DataFrame, fixture_raw_dir: Path, monkeypatch
) -> None:
    from ai_mcdm_invest.data import worked_example

    monkeypatch.setattr(worked_example, "MAX_PASSAGE_CHARS", 70)

    passages = worked_example_passages(filings, corpus_dir(fixture_raw_dir), "ROMO", 2015)

    risk = passages.set_index("item").loc["1A", "passage"]
    assert risk == "We have lost money for three years. We may need more capital. […]"


def test_an_example_from_the_test_years_is_refused(
    filings: pd.DataFrame, fixture_raw_dir: Path
) -> None:
    with pytest.raises(ValueError, match="fiscal 2017 is a test year"):
        worked_example_passages(filings, corpus_dir(fixture_raw_dir), "ROMO", 2017)


def test_an_example_without_10k_text_is_refused(
    filings: pd.DataFrame, fixture_raw_dir: Path
) -> None:
    with pytest.raises(ValueError, match="PAPA fiscal 2015 has no 10-K text"):
        worked_example_passages(filings, corpus_dir(fixture_raw_dir), "PAPA", 2015)
