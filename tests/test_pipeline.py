"""The rebuild command: raw data in, derived tables on disk."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data import join_report
from ai_mcdm_invest.data.join_report import JoinCoverageTooLow
from ai_mcdm_invest.data.raw_manifest import RawDataChanged
from ai_mcdm_invest.pipeline import run

FIXTURE_MANIFEST = Path(__file__).parent / "fixtures" / "raw_manifest.json"


def fixture_settings(raw_dir: Path, data_dir: Path) -> Settings:
    return Settings(raw_dir=raw_dir, data_dir=data_dir, manifest_path=FIXTURE_MANIFEST)


def test_rebuild_writes_the_clean_panel_as_parquet_and_csv(
    fixture_raw_dir: Path, tmp_path: Path
) -> None:
    run(fixture_settings(fixture_raw_dir, tmp_path))

    parquet = pd.read_parquet(tmp_path / "interim" / "kaggle_clean.parquet")
    csv = pd.read_csv(tmp_path / "interim" / "kaggle_clean.csv")
    assert len(parquet) == 127
    assert parquet["tail_loss"].sum() == 26
    pd.testing.assert_frame_equal(parquet, csv, check_dtype=False)


def test_raw_data_location_comes_from_one_setting(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_MCDM_RAW_DIR", str(tmp_path / "elsewhere"))

    assert Settings.from_env().raw_dir == tmp_path / "elsewhere"


def test_missing_raw_data_folder_names_the_setting(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="AI_MCDM_RAW_DIR"):
        run(fixture_settings(tmp_path / "nowhere", tmp_path))


def test_rebuild_writes_the_company_links(fixture_raw_dir: Path, tmp_path: Path) -> None:
    run(fixture_settings(fixture_raw_dir, tmp_path))

    links = pd.read_parquet(tmp_path / "interim" / "company_links.parquet")
    assert len(links) == 127
    assert set(links["link_method"]) == {"figure_match", "ticker", "unlinked"}
    assert (tmp_path / "interim" / "company_links.csv").exists()


def test_rebuild_stops_when_raw_data_differs_from_the_manifest(
    fixture_raw_dir: Path, tmp_path: Path
) -> None:
    raw_copy = tmp_path / "raw"
    shutil.copytree(fixture_raw_dir, raw_copy)
    (raw_copy / "sec" / "company_tickers.json").write_text("{}")

    with pytest.raises(RawDataChanged, match="company_tickers.json"):
        run(fixture_settings(raw_copy, tmp_path / "data"))

    assert not (tmp_path / "data").exists()  # nothing built from changed data


def test_rebuild_writes_the_joined_panel_and_the_join_report(
    fixture_raw_dir: Path, tmp_path: Path
) -> None:
    run(fixture_settings(fixture_raw_dir, tmp_path))

    panel = pd.read_parquet(tmp_path / "processed" / "panel.parquet")
    assert len(panel) == 22 * 5
    assert not panel.duplicated(["cik", "fiscal_year"]).any()
    assert (tmp_path / "processed" / "panel.csv").exists()
    report = (tmp_path / "reports" / "join_report.md").read_text()
    assert report.startswith("# Join report")


def test_rebuild_stops_before_the_joined_panel_when_coverage_is_low(
    fixture_raw_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(join_report, "MIN_COMPANY_COVERAGE", 0.9)  # fixture links 88%

    with pytest.raises(JoinCoverageTooLow, match="88.0% of companies"):
        run(fixture_settings(fixture_raw_dir, tmp_path))

    assert (tmp_path / "reports" / "join_report.md").exists()  # to see what went wrong
    assert not (tmp_path / "processed" / "panel.parquet").exists()
