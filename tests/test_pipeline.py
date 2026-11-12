"""The rebuild command: raw data in, derived tables on disk."""

from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.pipeline import run


def test_rebuild_writes_the_clean_panel_as_parquet_and_csv(
    fixture_raw_dir: Path, tmp_path: Path
) -> None:
    run(Settings(raw_dir=fixture_raw_dir, data_dir=tmp_path))

    parquet = pd.read_parquet(tmp_path / "interim" / "kaggle_clean.parquet")
    csv = pd.read_csv(tmp_path / "interim" / "kaggle_clean.csv")
    assert len(parquet) == 125
    assert parquet["tail_loss"].sum() == 25
    pd.testing.assert_frame_equal(parquet, csv, check_dtype=False)


def test_raw_data_location_comes_from_one_setting(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_MCDM_RAW_DIR", str(tmp_path / "elsewhere"))

    assert Settings.from_env().raw_dir == tmp_path / "elsewhere"


def test_missing_raw_data_folder_names_the_setting(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="AI_MCDM_RAW_DIR"):
        run(Settings(raw_dir=tmp_path / "nowhere", data_dir=tmp_path))
