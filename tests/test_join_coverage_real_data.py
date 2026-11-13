"""Slow acceptance check on the real raw data: the join keeps the prototype's coverage.

Kept out of CI (it needs the raw data); run with `make test-slow`.
"""

import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.join_report import check_join_coverage, coverage, join_steps
from ai_mcdm_invest.data.joined import join_panel
from ai_mcdm_invest.data.links import link_companies, load_ticker_ciks, load_xbrl_figures
from ai_mcdm_invest.data.panel import clean_panel, load_raw_panel
from ai_mcdm_invest.data.raw_manifest import load_manifest, verify_raw_files

pytestmark = pytest.mark.slow


def test_fiscal_2017_join_keeps_the_prototype_coverage() -> None:
    settings = Settings.from_env()
    if not settings.raw_panel_dir.is_dir():
        pytest.skip(f"real raw data not found in {settings.raw_dir}")
    verify_raw_files(settings.raw_dir, load_manifest(settings.manifest_path))
    clean = clean_panel(load_raw_panel(settings.raw_panel_dir))
    links = link_companies(
        clean,
        load_xbrl_figures(settings.raw_frames_dir),
        load_ticker_ciks(settings.ticker_list_path),
    )
    joined = join_panel(clean, links)
    steps = join_steps(clean, links, joined)

    check_join_coverage(steps)
    share = coverage(steps)
    # The prototype linked 87.6% of companies and 90.6% of tail losers.
    assert share["companies"] == pytest.approx(0.876, abs=0.01)
    assert share["tail_losers"] == pytest.approx(0.906, abs=0.01)
    assert not joined.duplicated(["cik", "fiscal_year"]).any()
