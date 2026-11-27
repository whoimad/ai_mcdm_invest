"""Test-year results: the "+ LLM" ablation row, the primary tests and the
tables read beside them (tickets 18 and 20 of issue #6, PREREGISTRATION.md).

Reads the rebuild's tables and the test-year going-concern run's
(`going_concern_test_filings` and `_chunks`); writes `test_year_*` tables to
`data/processed` and a summary to `data/reports/test_year_results.md`.

Run with `make results` or `uv run python -m ai_mcdm_invest.evaluation.results`
after `make all` and the test-year run. No model is called.

Tested: the likelihood-ratio tests of the pre-registration (section 4), one
added flag in a logistic model of tail loss on the ML score's logit and a
2018 dummy. Descriptive only: the ablation and its AUC gain, the share of
flagged companies already in ML's riskiest fifth, and the disagreements.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.evaluation.ablation import (
    FIFTH,
    KEY,
    POOLED,
    TEXT_SCORE_COLUMN,
    ablation_table,
    evaluation_universe,
)
from ai_mcdm_invest.llm.going_concern import CLEAR, FAILED, FIELDS, FLAGGED
from ai_mcdm_invest.llm.keyword_benchmark import KEYWORD_VERSION
from ai_mcdm_invest.ml.risk_score import ML_SCORE_COLUMN
from ai_mcdm_invest.tables import write_table

RUN_NAME = "going_concern_test"
# A universe row the test-year run has no result for.
NOT_READ = "not_read"
LLM, KEYWORD = "LLM", KEYWORD_VERSION
# The three hypotheses (pre-registration, section 1): the flag tested, and
# the flags already in the model beside the ML logit and the year dummy.
HYPOTHESES = {
    "H1": ("llm_flag", []),
    "H2": ("keyword_flag", []),
    "H3": ("llm_flag", ["keyword_flag"]),
}
FLAG_NAMES = {"llm_flag": LLM, "keyword_flag": KEYWORD}
ALL, NO_FALLBACK = "all", "no fallback reads"
# Disagreements shown in the summary; the table on disk has them all.
SHOWN_DISAGREEMENTS = 10

# The pre-registered significance level.
ALPHA = 0.05
# The ML probability is clipped this far from 0 and 1 before its logit is taken.
PROBABILITY_CLIP = 1e-6
# Newton's method stops when no coefficient moves more than this.
TOLERANCE = 1e-10
MAX_ITERATIONS = 100
# The 97.5% point of the standard normal, for a 95% interval.
Z_95 = 1.959963984540054


def likelihood_ratio_test(
    table: pd.DataFrame, flag: str, given: Sequence[str] = ()
) -> dict[str, float]:
    """The likelihood-ratio test of `flag` in a logistic model of tail loss.

    The base model is tail loss ~ logit(`tail_loss_probability`) + one dummy
    per test year after the first (pre-registration, section 4), plus the
    flags in `given`; the full model adds `flag`. The statistic is twice the
    gain in log-likelihood, read against a chi-squared with one degree of
    freedom. The odds ratio is the flag's, from the full model, with a 95%
    Wald interval. Every number is empty when either model cannot be fitted
    (a flag no company has, or one that separates tail losers perfectly).
    """
    y = table["tail_loss"].astype(float).to_numpy()
    base = _design(table, given)
    full = np.column_stack([base, table[flag].astype(float).to_numpy()])
    base_fit, full_fit = _fit_logistic(base, y), _fit_logistic(full, y)
    result = {"companies": len(table), "flagged": int(table[flag].astype(bool).sum())}
    if base_fit is None or full_fit is None:
        nan = float("nan")
        keys = ["lr_statistic", "p_value", "odds_ratio", "odds_ratio_low", "odds_ratio_high"]
        return result | dict.fromkeys(keys, nan)
    coefficients, log_likelihood, covariance = full_fit
    statistic = max(2 * (log_likelihood - base_fit[1]), 0.0)
    b, se = coefficients[-1], math.sqrt(covariance[-1, -1])
    return result | {
        "lr_statistic": statistic,
        # The chi-squared survival function with one degree of freedom.
        "p_value": math.erfc(math.sqrt(statistic / 2)),
        "odds_ratio": math.exp(b),
        "odds_ratio_low": math.exp(b - Z_95 * se),
        "odds_ratio_high": math.exp(b + Z_95 * se),
    }


def _design(table: pd.DataFrame, given: Sequence[str]) -> np.ndarray:
    """The base model's columns: a constant, the ML logit, the year dummies and `given`."""
    p = table["tail_loss_probability"].clip(PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    columns = [np.ones(len(table)), np.log(p / (1 - p)).to_numpy()]
    years = sorted(set(table["fiscal_year"]))
    columns += [(table["fiscal_year"] == year).to_numpy(float) for year in years[1:]]
    columns += [table[name].astype(float).to_numpy() for name in given]
    return np.column_stack(columns)


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Unpenalised logistic regression by Newton's method: the coefficients,
    the log-likelihood and the coefficients' covariance, or None when the fit
    does not converge (a column of one value, or perfect separation)."""
    beta = np.zeros(x.shape[1])
    for _ in range(MAX_ITERATIONS):
        p = 1 / (1 + np.exp(-(x @ beta)))
        information = x.T @ (x * (p * (1 - p))[:, None])
        try:
            step = np.linalg.solve(information, x.T @ (y - p))
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < TOLERANCE:
            break
    else:
        return None
    eta = x @ beta
    # log(1 + e^eta), written to stay finite for large eta.
    log_likelihood = float(np.sum(y * eta - np.logaddexp(0, eta)))
    p = 1 / (1 + np.exp(-eta))
    information = x.T @ (x * (p * (1 - p))[:, None])
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return None
    return beta, log_likelihood, covariance


def going_concern_criterion(filing_results: pd.DataFrame) -> pd.DataFrame:
    """The text criterion from the test-year run's filing table.

    One row per filing: its key, `accession`, `status`, `fallback_read`,
    `llm_flag` (empty when the extraction failed), `keyword_flag`, and
    `going_concern_score`, higher safer: clear 1, flagged 0, empty when failed.
    """
    read = filing_results["status"].isin([FLAGGED, CLEAR])
    flagged = filing_results["status"] == FLAGGED
    return filing_results[[*KEY, "accession", "status", "fallback_read"]].assign(
        llm_flag=flagged.astype("boolean").where(read, pd.NA),
        keyword_flag=filing_results["keyword_flagged"].astype(bool),
        **{TEXT_SCORE_COLUMN: (~flagged).astype(float).where(read)},
    )


def text_universe(
    universe: pd.DataFrame, ml_criterion: pd.DataFrame, criterion: pd.DataFrame
) -> pd.DataFrame:
    """The evaluation universe with the ML probability and the text criterion.

    Every universe row is kept; one the run has no result for has status
    `NOT_READ` and no flags, and is ranked on its other criteria.
    """
    table = universe.merge(
        ml_criterion[[*KEY, "tail_loss_probability"]], on=KEY, how="left", validate="one_to_one"
    ).merge(criterion, on=KEY, how="left", validate="one_to_one")
    table["status"] = table["status"].fillna(NOT_READ)
    table["fallback_read"] = table["fallback_read"].fillna(False).astype(bool)
    return table


def primary_tests(table: pd.DataFrame) -> pd.DataFrame:
    """The three pre-registered likelihood-ratio tests, pooled over the test
    years, on the companies whose extraction did not fail (pre-registration,
    section 2). `significant` is a p-value below `ALPHA`."""
    read = table[table["llm_flag"].notna()].reset_index(drop=True)
    read = read.assign(llm_flag=read["llm_flag"].astype(bool))
    rows = []
    for hypothesis, (flag, given) in HYPOTHESES.items():
        result = likelihood_ratio_test(read, flag, given)
        rows.append(
            {
                "hypothesis": hypothesis,
                "flag": FLAG_NAMES[flag],
                "given": ", ".join(["ML", *(FLAG_NAMES[name] for name in given)]),
                **result,
                "significant": bool(result["p_value"] < ALPHA),
            }
        )
    return pd.DataFrame(rows)


def riskiest_fifth(table: pd.DataFrame) -> pd.DataFrame:
    """How many flagged companies ML already puts in its riskiest fifth, per
    flag, per test year and pooled, with the flagged companies' tail-loss rate.

    The riskiest fifth is each year's lowest 20% of `ml_risk_score`, cut as
    the ablation cuts it, over the whole universe.
    """
    year = table["fiscal_year"]
    fifth = FIFTH * year.map(year.value_counts())
    in_fifth = table[ML_SCORE_COLUMN].groupby(year).rank() <= fifth
    samples = [(str(y), year == y) for y in sorted(set(year))] + [(POOLED, year.notna())]
    rows = []
    for flag, name in FLAG_NAMES.items():
        flagged = table[flag].fillna(False).astype(bool)
        for fiscal_year, in_sample in samples:
            marked = flagged & in_sample
            count = int(marked.sum())
            rows.append(
                {
                    "flag": name,
                    "fiscal_year": fiscal_year,
                    "flagged": count,
                    "in_riskiest_fifth": int((marked & in_fifth).sum()),
                    "share": float((marked & in_fifth).sum() / count) if count else math.nan,
                    "flagged_tail_loss_rate": float(table.loc[marked, "tail_loss"].mean())
                    if count
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def disagreements(table: pd.DataFrame, chunk_results: pd.DataFrame) -> pd.DataFrame:
    """The filings the LLM and kw-1 disagree on, with what each quoted.

    A failed extraction is not a disagreement (it is listed apart). `quote`
    and the facts are the LLM's on the chunk it flagged, else on the chunk
    kw-1 flagged; `keyword_match` is kw-1's first matching sentence. Riskiest
    by ML first.
    """
    read = table[table["llm_flag"].notna()]
    differ = read[read["llm_flag"].astype(bool) != read["keyword_flag"]]
    chunks = chunk_results.assign(
        _order=np.select(
            [chunk_results["status"] == FLAGGED, chunk_results["keyword_match"].notna()], [0, 1], 2
        )
    ).sort_values(["accession", "_order", "chunk"])
    chosen = chunks.drop_duplicates("accession")[["accession", "chunk", "item", "quote", *FIELDS]]
    keyword = (
        chunk_results.dropna(subset=["keyword_match"])
        .sort_values(["accession", "chunk"])
        .drop_duplicates("accession")[["accession", "keyword_match"]]
    )
    columns = ["ticker", *KEY, "accession", "fallback_read", "tail_loss", "next_year_return"]
    return (
        differ[[*columns, ML_SCORE_COLUMN]]
        .assign(
            flagged_by=np.where(differ["llm_flag"].astype(bool), f"{LLM} only", f"{KEYWORD} only")
        )
        .merge(chosen, on="accession", how="left", validate="one_to_one")
        .merge(keyword, on="accession", how="left", validate="one_to_one")
        .sort_values([ML_SCORE_COLUMN, *KEY])
        .reset_index(drop=True)
    )


def test_year_results(
    universe: pd.DataFrame,
    ml_criterion: pd.DataFrame,
    directions: pd.DataFrame,
    filing_results: pd.DataFrame,
    chunk_results: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Every results table, by name (written as `test_year_<name>`).

    `ablation` is the ablation on the whole universe and again without
    fallback reads (`sample`), SAW and TOPSIS both; `auc_gain` is "+ LLM"
    minus "+ ML", pooled, descriptive; `tests` the three likelihood-ratio
    tests; `riskiest_fifth`, `disagreements` and `failed` as named;
    `universe` the rows every one of them is measured on.
    """
    table = text_universe(universe, ml_criterion, going_concern_criterion(filing_results))
    ablation = pd.concat(
        [
            ablation_table(table, directions).assign(sample=ALL),
            ablation_table(table[~table["fallback_read"]], directions).assign(sample=NO_FALLBACK),
        ],
        ignore_index=True,
    )
    ablation = ablation[["sample", *ablation.columns.drop("sample")]]
    failed = table.loc[table["status"].isin([FAILED, NOT_READ]), ["ticker", *KEY, "status"]]
    return {
        "universe": table,
        "ablation": ablation,
        "auc_gain": auc_gain(ablation),
        "tests": primary_tests(table),
        "riskiest_fifth": riskiest_fifth(table),
        "disagreements": disagreements(table, chunk_results),
        "failed": failed.merge(table[[*KEY, "tail_loss", "keyword_flag"]], on=KEY),
    }


def auc_gain(ablation: pd.DataFrame) -> pd.DataFrame:
    """The pooled tail-loss AUC of "+ ML" and "+ LLM" and their difference,
    per sample, profile and method. Descriptive only: no interval."""
    pooled = ablation[ablation["fiscal_year"] == POOLED]
    wide = pooled.pivot_table(
        index=["sample", "profile", "method"], columns="layer", values="tail_loss_auc"
    )
    return (
        wide[["+ ML", "+ LLM"]]
        .assign(gain=lambda t: t["+ LLM"] - t["+ ML"])
        .rename(columns={"+ ML": "auc_ml", "+ LLM": "auc_llm"})
        .rename_axis(columns=None)
        .reset_index()
    )


def render_summary(results: dict[str, pd.DataFrame]) -> str:
    """The results as a short markdown page; every number is in the tables too."""
    table = results["universe"]
    counts = pd.crosstab(table["fiscal_year"], table["status"])
    counts["fallback_read"] = table.groupby("fiscal_year")["fallback_read"].sum()
    counts.loc["all"] = counts.sum()
    ablation = results["ablation"]
    pooled = ablation[(ablation["fiscal_year"] == POOLED)]
    columns = [
        "layer",
        "profile",
        "companies",
        "tail_loss_auc",
        "riskiest_fifth_tail_share",
        "safest_fifth_tail_share",
        "top_minus_bottom_return",
    ]

    def block(frame: pd.DataFrame) -> str:
        return f"```\n{frame.round(3).to_string(index=False)}\n```\n"

    def ablation_block(sample: str, method: str) -> str:
        rows = pooled[(pooled["sample"] == sample) & pooled["method"].isin([method, None])]
        return block(rows[columns])

    failed = results["failed"]
    shown = results["disagreements"].head(SHOWN_DISAGREEMENTS)
    shown_columns = ["ticker", "fiscal_year", "flagged_by", "tail_loss", "quote", "keyword_match"]
    tests = results["tests"][
        ["hypothesis", "flag", "given", "companies", "flagged", "lr_statistic", "p_value"]
        + ["odds_ratio", "odds_ratio_low", "odds_ratio_high", "significant"]
    ]
    return "\n".join(
        [
            "# Test-year results, fiscal 2017 and 2018 pooled",
            "",
            f"Evaluation universe: {len(table)} companies, {int(table['tail_loss'].sum())} "
            "tail losers. Going-concern run, by status:",
            "",
            block(counts.reset_index()),
            f"Left out of the tests (failed or not read, listed in `test_year_failed.csv`): "
            f"{len(failed)}, tail-loss rate {failed['tail_loss'].mean():.1%}."
            if len(failed)
            else "No extraction failed.",
            "",
            "## Primary tests (pre-registered)",
            "",
            "Likelihood-ratio test of one flag, 1 degree of freedom, alpha = "
            f"{ALPHA}. Base model: tail loss ~ logit(ML tail-loss probability) + a 2018 "
            "dummy (+ kw-1 flag for H3). Odds ratio of the flag with a 95% Wald interval.",
            "",
            block(tests),
            "## Ablation (descriptive; rows with ML are exploratory)",
            "",
            "Pooled, SAW:",
            "",
            ablation_block(ALL, "SAW"),
            'AUC gain of "+ LLM" over "+ ML" (point value, descriptive, no interval):',
            "",
            block(results["auc_gain"]),
            "Robustness, pooled, TOPSIS:",
            "",
            ablation_block(ALL, "TOPSIS"),
            "Robustness, pooled, SAW, without fallback reads:",
            "",
            ablation_block(NO_FALLBACK, "SAW"),
            "## Flagged companies already in ML's riskiest fifth (descriptive)",
            "",
            block(results["riskiest_fifth"]),
            f"## LLM and {KEYWORD} disagree (descriptive)",
            "",
            f"{len(results['disagreements'])} filings; the {len(shown)} ML finds riskiest "
            "are shown, all are in `test_year_disagreements.csv`.",
            "",
            block(shown[shown_columns]),
        ]
    )


def run(settings: Settings) -> None:
    """Read the rebuild's tables and the test-year run's, write the results.

    Stops with a message naming the missing table when the test-year run has
    not written its tables yet.
    """
    processed = settings.processed_dir
    paths = {part: processed / f"{RUN_NAME}_{part}.parquet" for part in ["filings", "chunks"]}
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise SystemExit(
            f"test-year results missing: {names}. Run the test-year going-concern run first "
            "(uv run python -m ai_mcdm_invest.llm.test_year_run)."
        )
    ml_criterion = pd.read_parquet(processed / "ml_criterion.parquet")
    directions = pd.read_parquet(processed / "criterion_directions.parquet")
    universe = evaluation_universe(
        pd.read_parquet(processed / "panel.parquet"),
        pd.read_parquet(processed / "ratio_criteria.parquet"),
        pd.read_parquet(processed / "filings.parquet"),
        pd.read_parquet(processed / "baselines.parquet"),
        ml_criterion,
    )
    results = test_year_results(
        universe,
        ml_criterion,
        directions,
        pd.read_parquet(paths["filings"]),
        pd.read_parquet(paths["chunks"]),
    )
    for name, table in results.items():
        if name != "universe":
            path = write_table(table, processed, f"test_year_{name}")
            print(f"{name}: {len(table)} rows, written to {path}")
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report = settings.reports_dir / "test_year_results.md"
    report.write_text(render_summary(results))
    print(f"summary: written to {report}")
    print(results["tests"].round(4).to_string(index=False))


def main() -> None:
    run(Settings.from_env())


if __name__ == "__main__":
    main()
