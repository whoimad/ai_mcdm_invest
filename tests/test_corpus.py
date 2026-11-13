"""EDGAR-CORPUS loader: 10-K records split by Item in, facts about each record out."""

from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.corpus import load_corpus


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    raw_dir = Path(__file__).parent / "fixtures" / "raw"
    return load_corpus(Settings(raw_dir=raw_dir).raw_corpus_dir, {1000, 1012, 1013, 1017})


def record(corpus: pd.DataFrame, cik: int, period_end: str) -> pd.Series:
    return corpus.set_index(["cik", "period_end"]).loc[(cik, pd.Timestamp(period_end))]


def test_keeps_only_the_requested_filers(corpus: pd.DataFrame) -> None:
    assert set(corpus["cik"]) == {1000, 1012, 1013, 1017}
    assert len(corpus) == 4 * 6  # one 10-K a year, periods ending 2013-2018


def test_period_comes_from_the_text_not_the_label(corpus: pd.DataFrame) -> None:
    # MIKE's year ending January 2017 is labelled 2016 in the corpus.
    mike = record(corpus, 1012, "2017-01-31")

    assert mike["corpus_year"] == 2016


def test_tokens_are_characters_over_four(corpus: pd.DataFrame) -> None:
    alfa = record(corpus, 1000, "2016-12-31")

    assert alfa["tokens_item_1"] == 500  # Item 1 is exactly 2,000 characters
    assert alfa["tokens_item_1A"] > 0
    assert alfa["tokens_total"] > alfa["tokens_item_1"] + alfa["tokens_item_1A"]


def test_stub_sections_count_as_absent(corpus: pd.DataFrame) -> None:
    # NOVM 2016: Item 1A reads "Not applicable." and Item 8 points to Item 15.
    novm = record(corpus, 1013, "2016-12-31")

    assert not novm["has_item_1A"]
    assert not novm["has_item_8"]
    assert novm["has_item_7"]
    assert novm["has_item_3"]  # short but real: "none of which is material"
    assert record(corpus, 1013, "2015-12-31")["has_item_1A"]


def test_going_concern_wording_is_located_by_item(corpus: pd.DataFrame) -> None:
    assert record(corpus, 1017, "2015-12-31")["gc_keyword_items"] == "1A;8"
    assert record(corpus, 1017, "2016-12-31")["gc_keyword_items"] == ""


def test_missing_corpus_folder_says_how_to_fetch(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make fetch"):
        load_corpus(tmp_path / "nowhere", {1000})
