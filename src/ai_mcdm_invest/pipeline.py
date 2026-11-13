"""The rebuild command: runs every stage in order, from raw data to tables.

Run with `make all` or `uv run python -m ai_mcdm_invest.pipeline`.
"""

import pandas as pd

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.links import link_companies, load_ticker_ciks, load_xbrl_figures
from ai_mcdm_invest.data.panel import clean_panel, load_raw_panel
from ai_mcdm_invest.tables import write_table


def run(settings: Settings) -> None:
    """Rebuild every derived table under `settings.data_dir`."""
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
