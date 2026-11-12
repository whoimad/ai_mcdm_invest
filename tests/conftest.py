from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.panel import load_raw_panel

FIXTURE_RAW_DIR = Path(__file__).parent / "fixtures" / "raw"


@pytest.fixture
def fixture_raw_dir() -> Path:
    """Raw-data folder of the fixture dataset (see fixtures/build_fixture.py)."""
    return FIXTURE_RAW_DIR


@pytest.fixture
def raw_panel() -> pd.DataFrame:
    return load_raw_panel(Settings(raw_dir=FIXTURE_RAW_DIR).raw_panel_dir)
