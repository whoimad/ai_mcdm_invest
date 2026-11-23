"""The rebuild command: runs every stage in order, from raw data to tables and figures.

Run with `make all` or `uv run python -m ai_mcdm_invest.pipeline`.
"""

import pandas as pd

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.criteria.ratios import learn_directions, score_ratios
from ai_mcdm_invest.data.corpus import load_corpus
from ai_mcdm_invest.data.filings import build_filings
from ai_mcdm_invest.data.join_report import (
    check_join_coverage,
    join_steps,
    render_join_report,
)
from ai_mcdm_invest.data.joined import join_panel
from ai_mcdm_invest.data.links import link_companies, load_ticker_ciks, load_xbrl_figures
from ai_mcdm_invest.data.panel import TEST_YEARS, clean_panel, load_raw_panel
from ai_mcdm_invest.data.raw_manifest import load_manifest, verify_raw_files
from ai_mcdm_invest.data.submissions import load_submissions
from ai_mcdm_invest.data.worked_example import worked_example_passages
from ai_mcdm_invest.evaluation.ablation import ablation_table, baselines, evaluation_universe
from ai_mcdm_invest.figures.common import write_figures
from ai_mcdm_invest.figures.draw import report_figures
from ai_mcdm_invest.ml.risk_score import ml_auc_by_year, ml_risk_criterion, ml_risk_scores
from ai_mcdm_invest.tables import write_table


def run(settings: Settings) -> None:
    """Rebuild every derived table and figure under `settings.data_dir`.

    Stops before writing anything if the raw files differ from the manifest,
    and before the joined panel if the join report shows too low a coverage.
    """
    verify_raw_files(settings.raw_dir, load_manifest(settings.manifest_path))
    raw = load_raw_panel(settings.raw_panel_dir)
    clean = clean_panel(raw)
    path = write_table(clean, settings.interim_dir, "kaggle_clean")
    print(f"clean panel: {len(raw)} raw rows -> {len(clean)} rows, written to {path}")
    by_year = clean.groupby("fiscal_year")["tail_loss"].agg(["size", "sum"])
    for fiscal_year, row in by_year.iterrows():
        print(f"  fiscal {fiscal_year}: {row['size']} companies, {row['sum']} tail losers")

    links = link_companies(
        clean,
        load_xbrl_figures(settings.raw_frames_dir),
        load_ticker_ciks(settings.ticker_list_path),
    )
    path = write_table(links, settings.interim_dir, "company_links")
    print(f"company links: written to {path}")
    print(_link_counts(links, clean).to_string())

    joined = join_panel(clean, links)
    steps = join_steps(clean, links, joined)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.reports_dir / "join_report.md"
    report_path.write_text(render_join_report(steps, links))
    print(f"join report: written to {report_path}")
    check_join_coverage(steps)
    path = write_table(joined, settings.processed_dir, "panel")
    print(f"joined panel: {len(joined)} rows, written to {path}")

    ciks = set(joined["cik"])
    filings = build_filings(
        joined,
        links,
        load_submissions(settings.submissions_path, ciks),
        load_corpus(settings.raw_corpus_dir, ciks),
    )
    path = write_table(filings, settings.processed_dir, "filings")
    print(f"filings table: {len(filings)} rows, written to {path}")
    print(_filing_counts(filings).to_string())
    print(_no_report_line(filings))

    directions = learn_directions(joined)
    path = write_table(directions, settings.processed_dir, "criterion_directions")
    print(f"criterion directions: written to {path}")
    print(
        directions[["criterion", "training_auc", "flipped_training_auc", "direction"]].to_string(
            index=False
        )
    )
    criteria = score_ratios(joined, directions)
    path = write_table(criteria, settings.processed_dir, "ratio_criteria")
    print(f"ratio criteria: {len(criteria)} rows, written to {path}")
    print(_too_few_ratios(criteria, joined).to_string())

    risk = ml_risk_scores(clean)
    path = write_table(risk, settings.processed_dir, "ml_risk")
    print(f"ML risk score: {len(risk)} rows, written to {path}")
    print("  exploratory: target and model family were chosen while looking at the test years")
    print(ml_auc_by_year(risk, clean).to_string())
    ml_criterion = ml_risk_criterion(risk, joined)
    path = write_table(ml_criterion, settings.processed_dir, "ml_criterion")
    print(f"ML criterion: {len(ml_criterion)} rows, written to {path}")

    baseline_table = baselines(joined)
    path = write_table(baseline_table, settings.processed_dir, "baselines")
    print(f"baselines: written to {path}")
    universe = evaluation_universe(joined, criteria, filings, baseline_table, ml_criterion)
    print(_universe_counts(universe).to_string())
    ablation = ablation_table(universe, directions)
    path = write_table(ablation, settings.processed_dir, "ablation")
    print(f"ablation table: {len(ablation)} rows, written to {path}")
    print(_ablation_summary(ablation).to_string(index=False))

    passages = worked_example_passages(filings, settings.raw_corpus_dir, *settings.worked_example)
    path = write_table(passages, settings.processed_dir, "worked_example")
    print(f"worked example: {len(passages)} passages, written to {path}")

    figures = report_figures(clean, links, joined, filings, passages)
    path = write_figures(figures, settings.figures_dir)
    print(f"figures: {len(figures)} written to {path.parent}")


def _link_counts(links: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Rows per fiscal year by link method, with ambiguous matches and tail losers."""
    table = links.merge(clean[["ticker", "fiscal_year", "tail_loss"]], on=["ticker", "fiscal_year"])
    counts = pd.crosstab(table["fiscal_year"], table["link_method"])
    counts["ambiguous"] = table[table["figure_match"] == "ambiguous"].groupby("fiscal_year").size()
    linked = table[table["link_method"] != "unlinked"]
    counts["tail_linked"] = linked.groupby("fiscal_year")["tail_loss"].sum()
    counts["tail_total"] = table.groupby("fiscal_year")["tail_loss"].sum()
    return counts.fillna(0).astype(int)


def _too_few_ratios(criteria: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    """Rows per fiscal year missing more than half of the ratios, with their tail losers."""
    too_few = criteria["too_few_ratios"]
    tail_loss = joined.loc[too_few, "tail_loss"]
    return (
        pd.DataFrame(
            {
                "rows": criteria.groupby("fiscal_year").size(),
                "too_few_ratios": too_few.groupby(criteria["fiscal_year"]).sum(),
                "their_tail_losers": tail_loss.groupby(criteria.loc[too_few, "fiscal_year"]).sum(),
            }
        )
        .fillna(0)
        .astype(int)
    )


def _universe_counts(universe: pd.DataFrame) -> pd.DataFrame:
    """Companies and tail losers per test year in the evaluation universe."""
    by_year = universe.groupby("fiscal_year")["tail_loss"]
    return pd.DataFrame({"evaluated": by_year.size(), "tail_losers": by_year.sum()})


def _ablation_summary(ablation: pd.DataFrame) -> pd.DataFrame:
    """The ablation's pooled SAW rows and baselines, rounded to read on screen."""
    pooled = ablation[(ablation["fiscal_year"] == "pooled") & (ablation["method"] != "TOPSIS")]
    columns = [
        "layer",
        "profile",
        "exploratory",
        "companies",
        "tail_loss_auc",
        "riskiest_fifth_tail_share",
        "safest_fifth_tail_share",
        "rank_correlation",
        "top_minus_bottom_return",
    ]
    return pooled[columns].round(3)


def _filing_counts(filings: pd.DataFrame) -> pd.DataFrame:
    """Rows per fiscal year with a 10-K, with its text, and with Items 1A, 7 and
    8; then the 10-Ks with no audit opinion text ("no report", left out of the
    evaluation universe), with their tail losers."""
    has_items = filings[["has_item_1A", "has_item_7", "has_item_8"]].fillna(False).all(axis=1)
    no_report = filings["no_report"].fillna(False).astype(bool)
    counts = filings.assign(
        rows=1,
        has_10k=filings["filing_date"].notna(),
        items_1a_7_8=has_items,
        no_report=no_report,
        no_report_tail_losers=no_report & filings["tail_loss"].astype(bool),
    )
    columns = ["rows", "has_10k", "in_corpus", "items_1a_7_8", "no_report", "no_report_tail_losers"]
    return counts.groupby("fiscal_year")[columns].sum()


def _no_report_line(filings: pd.DataFrame) -> str:
    """The training-year 10-Ks in the corpus with no audit opinion text (left
    out of the evaluation universe, decision 5.2) and their tail-loss rate,
    against that of every training-year 10-K in the corpus."""
    in_corpus = filings[~filings["fiscal_year"].isin(TEST_YEARS) & filings["in_corpus"]]
    no_report = in_corpus[in_corpus["no_report"].astype(bool)]
    years = f"fiscal {in_corpus['fiscal_year'].min()}-{in_corpus['fiscal_year'].max()}"
    return (
        f"no report, {years}: {len(no_report)} of {len(in_corpus)} 10-Ks in the corpus, "
        f"tail-loss rate {no_report['tail_loss'].mean():.1%} "
        f"against {in_corpus['tail_loss'].mean():.1%} overall"
    )


def main() -> None:
    run(Settings.from_env())


if __name__ == "__main__":
    main()
