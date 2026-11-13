"""The rebuild command: raw data in, derived tables on disk."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data import join_report
from ai_mcdm_invest.data.join_report import JoinCoverageTooLow
from ai_mcdm_invest.data.raw_manifest import RawDataChanged
from ai_mcdm_invest.figures.draw import draw_figures
from ai_mcdm_invest.pipeline import run

FIXTURE_MANIFEST = Path(__file__).parent / "fixtures" / "raw_manifest.json"
# ROMO's fiscal 2015 10-K has going-concern wording in Items 1A and 8.
FIXTURE_EXAMPLE = ("ROMO", 2015)


def fixture_settings(raw_dir: Path, data_dir: Path) -> Settings:
    return Settings(
        raw_dir=raw_dir,
        data_dir=data_dir,
        manifest_path=FIXTURE_MANIFEST,
        worked_example=FIXTURE_EXAMPLE,
    )


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


def test_worked_example_comes_from_one_setting(monkeypatch) -> None:
    monkeypatch.delenv("AI_MCDM_WORKED_EXAMPLE", raising=False)
    assert Settings.from_env().worked_example == ("CZR", 2014)

    monkeypatch.setenv("AI_MCDM_WORKED_EXAMPLE", "ROMO:2015")

    assert Settings.from_env().worked_example == ("ROMO", 2015)


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


def test_rebuild_writes_the_filings_table(fixture_raw_dir: Path, tmp_path: Path) -> None:
    run(fixture_settings(fixture_raw_dir, tmp_path))

    filings = pd.read_parquet(tmp_path / "processed" / "filings.parquet")
    panel = pd.read_parquet(tmp_path / "processed" / "panel.parquet")
    assert len(filings) == len(panel)
    assert filings["filing_date"].notna().sum() == len(panel) - 1  # QUBC 2017 filed none
    assert filings["in_corpus"].sum() == len(panel) - 2  # nor is PAPA 2015 in the corpus
    assert (tmp_path / "processed" / "filings.csv").exists()


def test_rebuild_stops_before_the_joined_panel_when_coverage_is_low(
    fixture_raw_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(join_report, "MIN_COMPANY_COVERAGE", 0.9)  # fixture links 88%

    with pytest.raises(JoinCoverageTooLow, match="88.0% of companies"):
        run(fixture_settings(fixture_raw_dir, tmp_path))

    assert (tmp_path / "reports" / "join_report.md").exists()  # to see what went wrong
    assert not (tmp_path / "processed" / "panel.parquet").exists()


def test_rebuild_writes_the_worked_example_passages(fixture_raw_dir: Path, tmp_path: Path) -> None:
    run(fixture_settings(fixture_raw_dir, tmp_path))

    passages = pd.read_parquet(tmp_path / "processed" / "worked_example.parquet")
    assert list(passages["item"]) == ["8", "1A", "3"]
    assert set(passages["ticker"]) == {"ROMO"}


STORYLINE_FIGURES = ["join_funnel", "leverage_tail_loss", "section_lengths", "going_concern_items"]
EXPLORATORY_FIGURES = [
    "companies_per_sector",
    "missing_values",
    "market_cap",
    "valuation_by_sector",
    "criteria_correlation",
    "return_distribution",
    "tail_loss_by_size_sector",
    "item1a_length",
    "filing_lag",
    "item1a_change",
    "worked_example",
]
FIGURES = STORYLINE_FIGURES + EXPLORATORY_FIGURES
# Figures not labelled with every fiscal year: filing lag against the outcome
# leaves out the test years (decision 5.1); the worked example is one company-year.
YEARS = {
    "filing_lag": "fiscal 2014–2016, returns in 2015–2017",
    "worked_example": "fiscal 2015, returns in 2016",
}


def test_rebuild_draws_every_figure_with_its_caption(fixture_raw_dir: Path, tmp_path: Path) -> None:
    run(fixture_settings(fixture_raw_dir, tmp_path))

    for name in FIGURES:
        assert (tmp_path / "figures" / f"{name}.pdf").exists()
        assert (tmp_path / "figures" / f"{name}.png").exists()
    captions = (tmp_path / "figures" / "captions.md").read_text()
    sections = dict(section.split("\n", 1) for section in captions.split("\n## ")[1:])
    assert list(sections) == FIGURES
    for name, caption in sections.items():
        assert "n = " in caption, name
        assert "Source:" in caption, name
        assert YEARS.get(name, "fiscal 2014–2018, returns in 2015–2019") in caption, name
    # The funnel starts from the whole clean panel: 127 company-years, 26 tail losers.
    assert "n = 127 company-years (26 tail losers)" in sections["join_funnel"]
    assert "ROMO" in sections["worked_example"]


def test_figures_are_redrawn_from_the_tables_on_disk(fixture_raw_dir: Path, tmp_path: Path) -> None:
    run(fixture_settings(fixture_raw_dir, tmp_path))
    shutil.rmtree(tmp_path / "figures")

    draw_figures(fixture_settings(tmp_path / "no_raw_data_needed", tmp_path))

    assert (tmp_path / "figures" / "captions.md").exists()
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == len(FIGURES)
