"""Raw-data manifest: the build runs only on the exact raw files it was recorded on."""

import shutil
from pathlib import Path

import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.raw_manifest import (
    RawDataChanged,
    raw_files_read,
    record_manifest,
    verify_raw_files,
)

FILES = ["2014 2018/2014_Financial_Data.csv", "sec/company_tickers.json"]


@pytest.fixture
def raw_copy(fixture_raw_dir: Path, tmp_path: Path) -> Path:
    """A writable copy of the fixture raw folder."""
    copy = tmp_path / "raw"
    shutil.copytree(fixture_raw_dir, copy)
    return copy


def only(raw_dir: Path, files: list[str]) -> Path:
    """Delete every file under `raw_dir` that is not in `files`."""
    for path in [p for p in raw_dir.rglob("*") if p.is_file()]:
        if path.relative_to(raw_dir).as_posix() not in files:
            path.unlink()
    return raw_dir


def test_unchanged_files_pass(raw_copy: Path) -> None:
    raw_dir = only(raw_copy, FILES)
    manifest = record_manifest(raw_dir, FILES)

    verify_raw_files(raw_dir, manifest)  # does not raise


def test_manifest_records_size_hash_and_date_of_each_file(raw_copy: Path) -> None:
    manifest = record_manifest(only(raw_copy, FILES), FILES)

    entry = manifest["files"]["sec/company_tickers.json"]
    assert entry["bytes"] == (raw_copy / "sec/company_tickers.json").stat().st_size
    assert len(entry["sha256"]) == 64
    assert len(entry["modified"]) == len("2026-11-12")


def test_a_changed_file_stops_the_build_and_is_named(raw_copy: Path) -> None:
    raw_dir = only(raw_copy, FILES)
    manifest = record_manifest(raw_dir, FILES)
    with (raw_dir / "sec/company_tickers.json").open("a") as file:
        file.write(" ")

    with pytest.raises(RawDataChanged, match="changed: sec/company_tickers.json"):
        verify_raw_files(raw_dir, manifest)


def test_a_missing_file_stops_the_build_and_is_named(raw_copy: Path) -> None:
    raw_dir = only(raw_copy, FILES)
    manifest = record_manifest(raw_dir, FILES)
    (raw_dir / "sec/company_tickers.json").unlink()

    with pytest.raises(RawDataChanged, match="missing: sec/company_tickers.json"):
        verify_raw_files(raw_dir, manifest)


def test_an_unlisted_file_beside_listed_ones_stops_the_build(raw_copy: Path) -> None:
    # The build reads whole folders (every XBRL frame), so an extra file would
    # enter the study unverified.
    raw_dir = only(raw_copy, FILES)
    manifest = record_manifest(raw_dir, FILES)
    (raw_dir / "sec/extra.json").write_text("{}")

    with pytest.raises(RawDataChanged, match="not in the manifest: sec/extra.json"):
        verify_raw_files(raw_dir, manifest)


def test_a_missing_raw_folder_names_the_setting(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="AI_MCDM_RAW_DIR"):
        verify_raw_files(tmp_path / "nowhere", {"files": {}})


def test_the_files_read_include_the_submissions_and_the_corpus(fixture_raw_dir: Path) -> None:
    files = raw_files_read(Settings(raw_dir=fixture_raw_dir))

    assert "sec/submissions.zip" in files
    assert "edgar_corpus/2016_train.jsonl" in files
