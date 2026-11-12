"""The rebuild command: runs every stage in order, from raw data to tables.

Run with `make all` or `uv run python -m ai_mcdm_invest.pipeline`.
"""

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.panel import clean_panel, load_raw_panel
from ai_mcdm_invest.tables import write_table


def run(settings: Settings) -> None:
    """Rebuild every derived table under `settings.data_dir`."""
    raw = load_raw_panel(settings.raw_panel_dir)
    clean = clean_panel(raw)
    path = write_table(clean, settings.interim_dir, "kaggle_clean")

    by_year = clean.groupby("fiscal_year")["tail_loss"].agg(["size", "sum"])
    print(f"clean panel: {len(raw)} raw rows -> {len(clean)} rows, written to {path}")
    for fiscal_year, row in by_year.iterrows():
        print(f"  fiscal {fiscal_year}: {row['size']} companies, {row['sum']} tail losers")


def main() -> None:
    run(Settings.from_env())


if __name__ == "__main__":
    main()
