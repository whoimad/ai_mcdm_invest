"""The screener page runs without exception on every year, method, profile and
layer switched off, with custom weights, with filters and with a card opened.

Kept out of CI (it needs the processed tables); run with `make test-slow`.
"""

from pathlib import Path

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

    # A row click stores the company's cik; the card follows it.
    table = load_table(Settings.from_env().processed_dir)[0]
    ticker = shown["Ticker"].iloc[-1]
    chosen = table[(table["ticker"] == ticker) & (table["fiscal_year"] == 2017)]
    app.session_state["card"] = int(chosen["cik"].iloc[0])
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
