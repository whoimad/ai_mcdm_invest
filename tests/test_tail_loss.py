"""The tail-loss label: the worst 20% of next-year returns within each year."""

import pandas as pd

from ai_mcdm_invest.data.panel import clean_panel

# The five lowest returns among each year's 25 fixture companies.
TAIL_LOSERS = {
    2014: {"ALFA", "BRVO", "CHRL", "DLTA", "ECHO"},
    2015: {"FXTR", "GOLF", "HOTL", "INDG", "JULT"},
    2016: {"KILO", "LIMA", "MIKE", "NOVM", "OSCR"},
    2017: {"PAPA", "QUBC", "ROMO", "SIER", "TNGO"},
    2018: {"UNFM", "VCTR", "WSKY", "XRAY", "YANK"},
}


def tail_losers(clean: pd.DataFrame, fiscal_year: int) -> set[str]:
    in_year = clean[clean["fiscal_year"] == fiscal_year]
    return set(in_year.loc[in_year["tail_loss"], "ticker"])


def test_marks_the_worst_fifth_of_each_year(raw_panel: pd.DataFrame) -> None:
    clean = clean_panel(raw_panel)

    for fiscal_year, expected in TAIL_LOSERS.items():
        assert tail_losers(clean, fiscal_year) == expected


def test_crash_year_does_not_label_every_company(raw_panel: pd.DataFrame) -> None:
    clean = clean_panel(raw_panel)
    crash = clean[clean["fiscal_year"] == 2015]

    assert (crash["next_year_return"] < 0).all()
    assert crash["tail_loss"].sum() == 5


def test_boom_year_still_has_tail_losers(raw_panel: pd.DataFrame) -> None:
    clean = clean_panel(raw_panel)
    boom = clean[clean["fiscal_year"] == 2016]

    assert (boom["next_year_return"] > 0).all()
    assert boom["tail_loss"].sum() == 5


def test_dropped_rows_do_not_shift_the_cut_off(raw_panel: pd.DataFrame) -> None:
    # ZZWO's -100% in fiscal 2017 is removed before the cut-off is taken, so
    # the fifth-worst valid company (TNGO, -38%) is still a tail loser.
    clean = clean_panel(raw_panel)

    assert "TNGO" in tail_losers(clean, 2017)
    assert len(tail_losers(clean, 2017)) == 5
