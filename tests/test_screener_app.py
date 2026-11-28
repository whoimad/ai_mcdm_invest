"""The screener page runs without exception on every year, method, profile and
layer switched off, with custom weights, with filters and with a card opened.

Kept out of CI (it needs the processed tables); run with `make test-slow`.
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.mcdm.ranking import PROFILES
from ai_mcdm_invest.screener import load_table

pytestmark = pytest.mark.slow

APP = str(Path(__file__).parents[1] / "app" / "screener.py")
LAYERS = ["ratios", "ml", "text"]


@pytest.fixture
def app() -> AppTest:
    processed = Settings.from_env().processed_dir
    if not (processed / "test_year_ablation.parquet").exists():
        pytest.skip(f"test-year results not found in {processed}")
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def key_of(table: pd.DataFrame, ticker: str, year: int) -> tuple[int, int]:
    row = table[(table["ticker"] == ticker) & (table["fiscal_year"] == year)].iloc[0]
    return int(row["cik"]), year


def open_card(at: AppTest, company: pd.Series) -> str:
    """Opens the company's card and returns its 10-K text tab's text."""
    at.button_group(key="year").set_value(int(company["fiscal_year"]))
    at.session_state["card"] = (int(company["cik"]), int(company["fiscal_year"]))
    at.text_input(key="search").set_value(company["ticker"])
    run(at)
    assert at.subheader[0].value.startswith(f"{company['ticker']} · ")
    tab = at.tabs[2]
    return "\n".join(e.value for e in [*tab.markdown, *tab.caption])


def run(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_every_year_method_profile_and_layer_off(app: AppTest) -> None:
    for year in [2017, 2018]:
        for method in ["SAW", "TOPSIS"]:
            for profile in PROFILES:
                for off in [None, *LAYERS]:
                    app.button_group(key="year").set_value(year)
                    app.button_group(key="method").set_value(method)
                    app.button_group(key="profile").set_value(profile)
                    app.button_group(key="layers").set_value([x for x in LAYERS if x != off])
                    run(app)
                    assert app.metric[0].label == "Companies shown"
                    # Only the study's own layer rows say the scores match it.
                    assert bool(app.success) == (off in [None, "text"])


def test_a_slider_switches_to_custom_and_reset_restores_the_profile(app: AppTest) -> None:
    app.button_group(key="profile").set_value("risk-averse")
    run(app)
    assert app.slider(key="weight_ml_risk").value == 40

    app.slider(key="weight_valuation").set_value(60)
    run(app)
    assert app.button_group(key="profile").value == "custom"
    assert "not part of the study's tests" in app.info[0].value

    app.button(key="reset").click()
    run(app)
    assert app.button_group(key="profile").value == "risk-averse"
    assert app.slider(key="weight_valuation").value == 10


def test_a_switched_off_layer_greys_its_slider(app: AppTest) -> None:
    app.button_group(key="layers").set_value(["ratios", "text"])
    run(app)

    assert app.slider(key="weight_ml_risk").disabled
    assert not app.slider(key="weight_valuation").disabled


def test_filters_hide_rows_and_a_card_opens_on_the_chosen_company(app: AppTest) -> None:
    app.toggle(key="reveal").set_value(True)
    app.multiselect(key="sectors").set_value(["Technology"])
    app.multiselect(key="statuses").set_value(["clear"])
    app.text_input(key="search").set_value("a")
    run(app)
    shown = app.dataframe[0].value
    assert 0 < len(shown) < 3900
    assert "Return over next calendar year (%)" in shown.columns

    # A click stores the company's (cik, fiscal year); the card follows it.
    table = load_table(Settings.from_env().processed_dir)[0]
    ticker = shown["Ticker"].iloc[-1]
    app.session_state["card"] = key_of(table, ticker, 2017)
    run(app)
    assert app.subheader[0].value.startswith(f"{ticker} · ")
    assert app.tabs[0].label == "Ratios"


def test_no_layer_and_all_zero_weights_are_explained(app: AppTest) -> None:
    app.button_group(key="layers").set_value([])
    run(app)
    assert "at least one layer" in app.info[0].value

    app.button_group(key="layers").set_value(["ratios"])
    run(app)
    for group in ["profitability", "valuation", "balance_sheet"]:
        app.slider(key=f"weight_{group}").set_value(0)
    run(app)
    assert "above zero" in app.warning[0].value


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return load_table(Settings.from_env().processed_dir)[0]


def test_a_flagged_card_shows_the_quotes_facts_link_and_kw_1(
    app: AppTest, table: pd.DataFrame
) -> None:
    company = table[(table["status"] == "flagged") & table["keyword_flag"].eq(True)].iloc[0]

    text = open_card(app, company)

    assert "Quoted from Item" in text
    assert "In the auditor's opinion: **yes**" in text
    assert "found in the 10-K text by code" in text
    assert "kw-1)**: flagged the filing." in text
    assert "disagree" not in text
    assert app.tabs[2].get("link_button")[0].proto.url == (
        f"https://www.sec.gov/Archives/edgar/data/{company['cik']}/"
        f"{company['accession'].replace('-', '')}/"
    )


@pytest.mark.parametrize(
    ("status", "keyword_flag", "line"),
    [
        ("clear", True, "the keywords flag the filing, the LLM does not"),
        ("flagged", False, "the LLM flags the filing, the keywords do not"),
    ],
)
def test_a_disagreement_is_said(
    app: AppTest, table: pd.DataFrame, status: str, keyword_flag: bool, line: str
) -> None:
    company = table[(table["status"] == status) & table["keyword_flag"].eq(keyword_flag)].iloc[0]

    assert line in open_card(app, company)


def test_the_card_is_one_of_the_rows_shown(app: AppTest, table: pd.DataFrame) -> None:
    # A company chosen, then hidden by the filter: the card falls back to the first row.
    app.session_state["card"] = key_of(table, "CSX", 2017)
    run(app)
    assert app.subheader[0].value.startswith("CSX · ")

    app.multiselect(key="statuses").set_value(["flagged"])
    run(app)
    first = app.dataframe[0].value["Ticker"].iloc[0]
    assert app.subheader[0].value.startswith(f"{first} · ")
    assert any("not among the rows shown" in c.value for c in app.caption)

    app.text_input(key="search").set_value("no such company")
    run(app)
    assert not app.subheader
    assert "No company matches" in app.info[0].value


def test_tail_loss_values_are_in_the_table(app: AppTest) -> None:
    assert app.dataframe[0].value["Tail-loss probability (%)"].notna().mean() > 0.9
