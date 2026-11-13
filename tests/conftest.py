from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.links import link_companies, load_ticker_ciks, load_xbrl_figures
from ai_mcdm_invest.data.panel import clean_panel, load_raw_panel

FIXTURE_RAW_DIR = Path(__file__).parent / "fixtures" / "raw"


@pytest.fixture
def fixture_raw_dir() -> Path:
    """Raw-data folder of the fixture dataset (see fixtures/build_fixture.py)."""
    return FIXTURE_RAW_DIR


@pytest.fixture
def raw_panel() -> pd.DataFrame:
    return load_raw_panel(Settings(raw_dir=FIXTURE_RAW_DIR).raw_panel_dir)


@pytest.fixture
def clean_fixture_panel(raw_panel: pd.DataFrame) -> pd.DataFrame:
    return clean_panel(raw_panel)


@pytest.fixture
def fixture_links(clean_fixture_panel: pd.DataFrame) -> pd.DataFrame:
    """The fixture's company links, indexed by ticker and fiscal year."""
    settings = Settings(raw_dir=FIXTURE_RAW_DIR)
    links = link_companies(
        clean_fixture_panel,
        load_xbrl_figures(settings.raw_frames_dir),
        load_ticker_ciks(settings.ticker_list_path),
    )
    return links.set_index(["ticker", "fiscal_year"])
