"""Join report: how many companies and tail losers each join step keeps or loses."""

import pandas as pd
import pytest

from ai_mcdm_invest.data.join_report import (
    JoinCoverageTooLow,
    check_join_coverage,
    join_steps,
    render_join_report,
)
from ai_mcdm_invest.data.joined import join_panel


@pytest.fixture
def steps(clean_fixture_panel: pd.DataFrame, fixture_links: pd.DataFrame) -> pd.DataFrame:
    links = fixture_links.reset_index()
    return join_steps(clean_fixture_panel, links, join_panel(clean_fixture_panel, links))


def counts(steps: pd.DataFrame, fiscal_year: int, column: str) -> dict[str, int]:
    return steps.loc[fiscal_year, column].to_dict()


def test_counts_companies_at_each_join_step(steps: pd.DataFrame) -> None:
    assert counts(steps, 2014, "companies") == {
        "clean panel": 25,
        "linked by ticker": 20,  # all but ALFA (reused), BRVO, CHRL, DLTA, HOTL
        "recovered by figures": 2,  # ALFA, BRVO
        "unlinked": 3,  # CHRL, DLTA, HOTL
        "duplicates dropped": 0,
        "joined panel": 22,
    }


def test_counts_tail_losers_kept_and_lost_at_each_step(steps: pd.DataFrame) -> None:
    # 2014's tail losers: ALFA, BRVO, CHRL, DLTA, ECHO.
    assert counts(steps, 2014, "tail_losers") == {
        "clean panel": 5,
        "linked by ticker": 1,  # ECHO
        "recovered by figures": 2,  # ALFA, BRVO
        "unlinked": 2,  # CHRL, DLTA
        "duplicates dropped": 0,
        "joined panel": 3,
    }


def test_duplicate_share_classes_are_counted_as_dropped(steps: pd.DataFrame) -> None:
    # LIMA and LIMAP collapse into LIMAK in 2018; LIMAP was a tail loser.
    assert steps.loc[(2018, "clean panel"), "companies"] == 27
    assert steps.loc[(2018, "duplicates dropped")].to_dict() == {
        "companies": 2,
        "tail_losers": 1,
    }
    assert steps.loc[(2018, "joined panel"), "companies"] == 22


def fiscal_2017_steps(companies: int, tail_losers: int) -> pd.DataFrame:
    """Steps where 1000 companies (200 tail losers) are split into linked and unlinked."""
    index = pd.MultiIndex.from_product([[2017], ["clean panel", "unlinked"]])
    return pd.DataFrame(
        {
            "companies": [1000, 1000 - companies],
            "tail_losers": [200, 200 - tail_losers],
        },
        index=index.set_names(["fiscal_year", "step"]),
    )


def test_coverage_near_the_prototype_passes(steps: pd.DataFrame) -> None:
    check_join_coverage(steps)  # fixture: 22 of 25 companies, 5 of 5 tail losers
    check_join_coverage(fiscal_2017_steps(companies=876, tail_losers=181))


def test_build_stops_when_too_few_companies_are_linked() -> None:
    with pytest.raises(JoinCoverageTooLow, match="80.0% of companies"):
        check_join_coverage(fiscal_2017_steps(companies=800, tail_losers=181))


def test_build_stops_when_too_few_tail_losers_are_linked() -> None:
    with pytest.raises(JoinCoverageTooLow, match="80.0% of tail losers"):
        check_join_coverage(fiscal_2017_steps(companies=876, tail_losers=160))


def sample_section(report: str) -> str:
    return report.split("## Recovered matches")[1]


def test_report_shows_each_step_per_year(steps: pd.DataFrame, fixture_links: pd.DataFrame) -> None:
    report = render_join_report(steps, fixture_links.reset_index())

    assert "| fiscal year | clean panel | linked by ticker | recovered by figures |" in report
    # LIMAK and LIMAP are not in today's ticker list, so their figures recover them.
    assert "| 2018 | 27 | 20 | 4 | 3 | 2 | 22 |" in report  # companies
    assert "| 2018 | 6 | 5 | 1 | 0 | 1 | 5 |" in report  # tail losers
    assert "fiscal 2017: 88.0% of companies and 100.0% of tail losers linked" in report


def test_report_samples_recovered_matches_only(
    steps: pd.DataFrame, fixture_links: pd.DataFrame
) -> None:
    sample = sample_section(render_join_report(steps, fixture_links.reset_index(), sample_size=4))

    rows = [line for line in sample.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) == 1 + 4  # header and four matches
    assert all(
        row.split(" | ")[0] in {"| ALFA", "| BRVO", "| LIMAK", "| LIMAP"} for row in rows[1:]
    )


def test_report_sample_is_the_same_on_every_build(
    steps: pd.DataFrame, fixture_links: pd.DataFrame
) -> None:
    links = fixture_links.reset_index()

    assert render_join_report(steps, links, sample_size=4) == render_join_report(
        steps, links.sample(frac=1, random_state=1), sample_size=4
    )
