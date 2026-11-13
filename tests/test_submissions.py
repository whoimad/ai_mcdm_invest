"""SEC submissions loader: the bulk submissions file in, one row per annual-report filing out."""

from pathlib import Path

import pandas as pd

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.submissions import load_submissions


def fixture_submissions(raw_dir: Path, ciks: set[int]) -> pd.DataFrame:
    return load_submissions(Settings(raw_dir=raw_dir).submissions_path, ciks)


def test_loads_10ks_and_late_filing_notices_of_the_requested_filers(
    fixture_raw_dir: Path,
) -> None:
    filings = fixture_submissions(fixture_raw_dir, {1000, 1014})

    assert set(filings["cik"]) == {1000, 1014}
    assert set(filings["form"]) == {"10-K", "NT 10-K"}  # no 10-Q, no 10-K/A
    oscr = filings[filings["cik"] == 1014].set_index("form")
    assert oscr.loc["NT 10-K", "filing_date"] == pd.Timestamp("2017-03-16")
    assert oscr.loc["NT 10-K", "report_date"] == pd.Timestamp("2016-12-31")


def test_reads_the_older_pages_of_a_filer(fixture_raw_dir: Path) -> None:
    # MIKE's filings before 2016 sit in a second file of the bulk archive.
    mike = fixture_submissions(fixture_raw_dir, {1012})

    assert sorted(mike["report_date"].dt.year) == list(range(2013, 2019))
    first = mike[mike["report_date"] == pd.Timestamp("2013-01-31")]
    assert first["filing_date"].item() == pd.Timestamp("2013-04-15")
    assert first["accession"].item().startswith("00001012-13-")


def test_filer_missing_from_the_archive_has_no_rows(fixture_raw_dir: Path) -> None:
    assert fixture_submissions(fixture_raw_dir, {424242}).empty
