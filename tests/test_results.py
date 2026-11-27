"""Test-year results: the "+ LLM" row, the likelihood-ratio tests and the tables
read beside them. Simulated data and the fixture only; no model is called."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.evaluation.results import (
    disagreements,
    going_concern_criterion,
    likelihood_ratio_test,
    primary_tests,
    riskiest_fifth,
    run,
)
from ai_mcdm_invest.pipeline import run as rebuild


def simulated(seed: int, flag_log_odds: float, n: int = 6000) -> pd.DataFrame:
    """Two test years of companies with an ML probability and a flag that
    raises the log-odds of a tail loss by `flag_log_odds` beyond the ML score.
    Flags go mostly to companies ML already finds risky, as in the pre-tests;
    2018 has a higher base rate than 2017, which the year dummy absorbs."""
    rng = np.random.default_rng(seed)
    ml_logit = rng.normal(-1.5, 1.0, n)
    fiscal_year = rng.choice([2017, 2018], n)
    flag = rng.random(n) < 1 / (1 + np.exp(-(ml_logit - 2.5)))
    log_odds = ml_logit + 0.4 * (fiscal_year == 2018) + flag_log_odds * flag
    return pd.DataFrame(
        {
            "fiscal_year": fiscal_year,
            "tail_loss_probability": 1 / (1 + np.exp(-ml_logit)),
            "llm_flag": flag,
            "tail_loss": rng.random(n) < 1 / (1 + np.exp(-log_odds)),
        }
    )


def test_the_likelihood_ratio_test_recovers_a_planted_flag_effect() -> None:
    result = likelihood_ratio_test(simulated(seed=1, flag_log_odds=math.log(3)), "llm_flag")

    assert result["p_value"] < 0.05
    assert result["odds_ratio_low"] < 3 < result["odds_ratio_high"]
    assert result["companies"] == 6000


def test_the_likelihood_ratio_test_finds_no_effect_under_the_null() -> None:
    result = likelihood_ratio_test(simulated(seed=2, flag_log_odds=0.0), "llm_flag")

    assert result["p_value"] > 0.05
    assert result["odds_ratio_low"] < 1 < result["odds_ratio_high"]


def test_the_llm_flag_adds_nothing_beyond_a_benchmark_that_holds_the_same_information() -> None:
    # Only the benchmark's flag carries the effect; the LLM's agrees with it
    # but for a few companies chosen at random.
    table = simulated(seed=3, flag_log_odds=math.log(3)).rename(
        columns={"llm_flag": "keyword_flag"}
    )
    swapped = np.random.default_rng(3).random(len(table)) < 0.01
    table["llm_flag"] = table["keyword_flag"] ^ swapped

    alone = likelihood_ratio_test(table, "llm_flag")
    beyond = likelihood_ratio_test(table, "llm_flag", given=["keyword_flag"])

    assert alone["p_value"] < 0.05
    assert beyond["p_value"] > 0.05


def test_a_flag_no_company_has_gives_an_empty_test_not_an_error() -> None:
    table = simulated(seed=4, flag_log_odds=0.0).assign(llm_flag=False)

    result = likelihood_ratio_test(table, "llm_flag")

    assert result["flagged"] == 0
    assert math.isnan(result["p_value"])


def filing_results(statuses: list[str], keyword: list[bool]) -> pd.DataFrame:
    """The test-year run's filing table, one 2017 filing per status."""
    n = len(statuses)
    return pd.DataFrame(
        {
            "cik": range(1, n + 1),
            "fiscal_year": 2017,
            "accession": [f"000-{i}" for i in range(1, n + 1)],
            "status": statuses,
            "fallback_read": False,
            "chunks": 1,
            "seconds": 1.0,
            "keyword_flagged": keyword,
            "keyword_version": "kw-1",
        }
    )


def test_the_text_criterion_scores_clear_1_flagged_0_and_a_failed_extraction_nothing() -> None:
    table = going_concern_criterion(
        filing_results(["clear", "flagged", "extraction_failed"], [False, True, True])
    )

    assert table["going_concern_score"].tolist()[:2] == [1.0, 0.0]
    assert pd.isna(table["going_concern_score"].iloc[2])
    assert table["llm_flag"].tolist()[:2] == [False, True]
    assert pd.isna(table["llm_flag"].iloc[2])
    assert table["keyword_flag"].tolist() == [False, True, True]


def test_the_three_tests_leave_failed_extractions_out() -> None:
    table = simulated(seed=5, flag_log_odds=math.log(3))
    table["keyword_flag"] = table["llm_flag"]
    table["llm_flag"] = table["llm_flag"].astype("boolean")
    table.loc[:9, "llm_flag"] = pd.NA

    tests = primary_tests(table).set_index("hypothesis")

    assert tests.index.tolist() == ["H1", "H2", "H3"]
    assert set(tests["companies"]) == {len(table) - 10}
    assert tests.loc["H3", "given"] == "ML, kw-1"
    assert tests.loc["H1", "significant"]
    # The two flags are one and the same: the LLM's adds nothing beyond kw-1.
    assert math.isnan(tests.loc["H3", "p_value"]) or not tests.loc["H3", "significant"]


def test_the_share_of_flagged_companies_already_in_ml_s_riskiest_fifth() -> None:
    # Ten companies; ML puts those scored 0.1 and 0.2 in the riskiest fifth.
    # The LLM flags them and the one scored 0.9; kw-1 flags only the first.
    table = pd.DataFrame(
        {
            "fiscal_year": 2017,
            "ml_risk_score": [i / 10 for i in range(1, 11)],
            "tail_loss": [True, True] + [False] * 8,
            "llm_flag": pd.array([True, True] + [False] * 6 + [True, False], dtype="boolean"),
            "keyword_flag": [True] + [False] * 9,
        }
    )

    shares = riskiest_fifth(table).set_index(["flag", "fiscal_year"])

    assert shares.loc[("LLM", "pooled"), "flagged"] == 3
    assert shares.loc[("LLM", "pooled"), "in_riskiest_fifth"] == 2
    assert shares.loc[("LLM", "pooled"), "share"] == pytest.approx(2 / 3)
    assert shares.loc[("LLM", "pooled"), "flagged_tail_loss_rate"] == pytest.approx(2 / 3)
    assert shares.loc[("kw-1", "2017"), "share"] == 1.0


def test_the_disagreements_are_the_filings_one_flag_alone_marks_with_their_quotes() -> None:
    filings = filing_results(
        ["flagged", "clear", "flagged", "clear", "extraction_failed"],
        [False, True, True, False, True],
    )
    table = going_concern_criterion(filings).assign(
        ticker=list("ABCDE"), tail_loss=False, next_year_return=0.0, ml_risk_score=0.5
    )
    chunks = filings[["cik", "fiscal_year", "accession"]].assign(
        chunk=0,
        item="8",
        status=["flagged", "clear", "flagged", "clear", "extraction_failed"],
        quote=["raise substantial doubt", "", "doubt", "", None],
        auditor_opinion=[True, False, True, False, None],
        this_company=[True, False, True, False, None],
        current_year=[True, False, True, False, None],
        alleviated=False,
        doubt_expressed=[True, False, True, False, None],
        keyword_match=[None, "substantial doubt about its ability", "doubt", None, "x"],
    )

    table = disagreements(table, chunks).set_index("ticker")

    # C is flagged by both and D by neither; E failed, so it is listed apart.
    assert table.index.tolist() == ["A", "B"]
    assert table.loc["A", "flagged_by"] == "LLM only"
    assert table.loc["A", "quote"] == "raise substantial doubt"
    assert table.loc["B", "flagged_by"] == "kw-1 only"
    assert table.loc["B", "keyword_match"] == "substantial doubt about its ability"


def going_concern_tables(processed: Path) -> None:
    """Fake test-year results for every test-year filing of the fixture: every
    fourth flagged by the LLM, every third by kw-1, one failed."""
    filings = pd.read_parquet(processed / "filings.parquet")
    test = filings[(filings["fiscal_year"] >= 2017) & filings["in_corpus"]].reset_index(drop=True)
    status = np.where(test.index % 4 == 0, "flagged", "clear").astype(object)
    status[1] = "extraction_failed"
    results = test[["cik", "fiscal_year", "accession"]].assign(
        status=status,
        fallback_read=test.index % 5 == 0,
        chunks=1,
        seconds=1.0,
        keyword_flagged=test.index % 3 == 0,
        keyword_version="kw-1",
    )
    flagged = results["status"] == "flagged"
    chunks = results[["cik", "fiscal_year", "accession"]].assign(
        chunk=0,
        item="8",
        status=results["status"],
        quote=np.where(flagged, "substantial doubt", ""),
        auditor_opinion=flagged,
        this_company=flagged,
        current_year=flagged,
        alleviated=False,
        doubt_expressed=flagged,
        model_version="fake",
        prompt_version="gc-2",
        keyword_match=results["keyword_flagged"].map({True: "substantial doubt", False: None}),
    )
    results.to_parquet(processed / "going_concern_test_filings.parquet", index=False)
    chunks.to_parquet(processed / "going_concern_test_chunks.parquet", index=False)


def test_on_the_fixture_every_layer_and_baseline_is_measured_on_the_same_companies(
    fixture_raw_dir: Path, tmp_path: Path
) -> None:
    settings = Settings(
        raw_dir=fixture_raw_dir,
        data_dir=tmp_path,
        manifest_path=Path(__file__).parent / "fixtures" / "raw_manifest.json",
        worked_example=("ROMO", 2015),
    )
    rebuild(settings)
    going_concern_tables(settings.processed_dir)

    run(settings)

    ablation = pd.read_parquet(settings.processed_dir / "test_year_ablation.parquet")
    everyone = ablation[ablation["sample"] == "all"]
    assert set(everyone["layer"]) == {
        "ratios only",
        "+ ML",
        "+ LLM",
        "size baseline",
        "last year's price move baseline",
    }
    # As in the rebuild's own ablation: 21 filers in 2017 and 22 in 2018.
    by_year = everyone.groupby("fiscal_year")["companies"].unique()
    assert by_year["2017"].tolist() == [21]
    assert by_year["2018"].tolist() == [22]
    no_fallback = ablation[ablation["sample"] == "no fallback reads"]
    by_year = no_fallback.groupby("fiscal_year")["companies"].unique()
    assert all(len(companies) == 1 for companies in by_year)
    assert no_fallback["companies"].max() < everyone["companies"].max()
    tests = pd.read_parquet(settings.processed_dir / "test_year_tests.parquet")
    assert set(tests["companies"]) == {43 - 1}  # the failed extraction left out
    summary = (settings.reports_dir / "test_year_results.md").read_text()
    assert "Likelihood-ratio test" in summary
    assert "descriptive" in summary


def test_without_the_test_year_tables_the_results_stop_and_say_which(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="going_concern_test_filings"):
        run(Settings(data_dir=tmp_path))
