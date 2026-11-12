"""Writing derived tables: Parquet for the code, a CSV copy to open by hand."""

from pathlib import Path

import pandas as pd


def write_table(table: pd.DataFrame, folder: Path, name: str) -> Path:
    """Write `table` as `<folder>/<name>.parquet` and `.csv`; return the Parquet path."""
    folder.mkdir(parents=True, exist_ok=True)
    parquet_path = folder / f"{name}.parquet"
    table.to_parquet(parquet_path, index=False)
    table.to_csv(folder / f"{name}.csv", index=False)
    return parquet_path
