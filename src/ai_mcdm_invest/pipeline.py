"""The rebuild command: runs every stage in order, from raw data to tables.

Run with `make all` or `uv run python -m ai_mcdm_invest.pipeline`.
"""

import pandas as pd

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.join_report import (
    check_join_coverage,
    join_steps,
    render_join_report,
)
from ai_mcdm_invest.data.joined import join_panel
from ai_mcdm_invest.data.links import link_companies, load_ticker_ciks, load_xbrl_figures
from ai_mcdm_invest.data.panel import clean_panel, load_raw_panel
from ai_mcdm_invest.data.raw_manifest import load_manifest, verify_raw_files
from ai_mcdm_invest.tables import write_table


def run(settings: Settings) -> None:
    """Rebuild every derived table under `settings.data_dir`.

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


def _link_counts(links: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Rows per fiscal year by link method, with ambiguous matches and tail losers."""
    table = links.merge(clean[["ticker", "fiscal_year", "tail_loss"]], on=["ticker", "fiscal_year"])
    counts = pd.crosstab(table["fiscal_year"], table["link_method"])
    counts["ambiguous"] = table[table["figure_match"] == "ambiguous"].groupby("fiscal_year").size()
    linked = table[table["link_method"] != "unlinked"]
    counts["tail_linked"] = linked.groupby("fiscal_year")["tail_loss"].sum()
    counts["tail_total"] = table.groupby("fiscal_year")["tail_loss"].sum()
    return counts.fillna(0).astype(int)


def main() -> None:
    run(Settings.from_env())


if __name__ == "__main__":
    main()
