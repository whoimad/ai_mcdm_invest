"""Clean panel stage: the raw fundamentals panel as one row per ticker and year.

The raw panel is one CSV per fiscal year. Each has the ticker as an unnamed
first column, about 220 fundamentals, a `Sector` column, the next calendar
year's price change in `<year + 1> PRICE VAR [%]`, and a `Class` column.
"""

from pathlib import Path

import pandas as pd

FISCAL_YEARS = range(2014, 2019)
# Held out for the test (decision 5.1): rules are developed on the years before.
TEST_YEARS = range(2017, 2019)

# A next-year return above this (in %) is almost surely a split or data error.
MAX_PLAUSIBLE_RETURN = 1000.0
# A price cannot fall by 100% or more.
MIN_PLAUSIBLE_RETURN = -100.0
# The tail-loss label marks this share of each year's lowest next-year returns.
TAIL_SHARE = 0.20

KEY_COLUMNS = ["ticker", "fiscal_year"]
RAW_REQUIRED_COLUMNS = [*KEY_COLUMNS, "Sector", "Class", "next_year_return"]
NON_FUNDAMENTAL_COLUMNS = [*KEY_COLUMNS, "sector", "next_year_return", "tail_loss"]


def load_raw_panel(raw_panel_dir: Path) -> pd.DataFrame:
    """Stack the yearly raw files, unchanged apart from the layout.

    Adds `ticker` and `fiscal_year` and gives the year-specific return column
    the common name `next_year_return`. No row or value is altered.
    """
    if not raw_panel_dir.is_dir():
        raise FileNotFoundError(
            f"raw panel folder not found: {raw_panel_dir}. "
            "Set AI_MCDM_RAW_DIR to the folder that contains it."
        )
    frames = []
    for year in FISCAL_YEARS:
        frame = pd.read_csv(raw_panel_dir / f"{year}_Financial_Data.csv")
        ticker_column = frame.columns[0]  # unnamed in the raw files
        frame = frame.rename(
            columns={ticker_column: "ticker", f"{year + 1} PRICE VAR [%]": "next_year_return"}
        )
        frames.append(pd.concat([frame, pd.Series(year, frame.index, name="fiscal_year")], axis=1))
    return pd.concat(frames, ignore_index=True)


def clean_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the stacked raw panel into the clean panel.

    One row per ticker and fiscal year, with every fundamental kept under its
    raw name, plus `sector`, `next_year_return` (in %) and `tail_loss`.

    Rows without a next-year return, or with an implausible one, are dropped:
    they cannot be labelled. The raw `Class` column is dropped because it is
    derived from the next-year return and would leak the outcome into the
    fundamentals. Rows with missing fundamentals are kept.
    """
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"raw panel lacks required columns: {missing}")
    unlabelled_years = raw.groupby("fiscal_year")["next_year_return"].count().eq(0)
    if unlabelled_years.any():
        years = unlabelled_years[unlabelled_years].index.tolist()
        raise ValueError(f"raw panel has no next-year return at all for fiscal {years}")

    duplicated = raw.duplicated(KEY_COLUMNS)
    if duplicated.any():
        examples = raw.loc[duplicated, KEY_COLUMNS].head().to_dict("records")
        raise ValueError(f"raw panel repeats a ticker within a fiscal year: {examples}")

    panel = raw.rename(columns={"Sector": "sector"}).drop(columns="Class")
    returns = panel["next_year_return"]
    plausible = (returns > MIN_PLAUSIBLE_RETURN) & (returns <= MAX_PLAUSIBLE_RETURN)
    panel = panel[returns.notna() & plausible].copy()
    panel["tail_loss"] = tail_loss_label(panel)

    fundamentals = [c for c in panel.columns if c not in NON_FUNDAMENTAL_COLUMNS]
    panel = panel[[*KEY_COLUMNS, "sector", *fundamentals, "next_year_return", "tail_loss"]]
    return panel.sort_values(KEY_COLUMNS, ignore_index=True)


def tail_loss_label(panel: pd.DataFrame) -> pd.Series:
    """Whether a row is in the worst `TAIL_SHARE` of its own year's returns.

    The cut-off is taken within each fiscal year, so the label says which
    companies were fragile and not which year the market fell.
    """
    returns = panel["next_year_return"]
    cutoff = returns.groupby(panel["fiscal_year"]).transform("quantile", TAIL_SHARE)
    return returns <= cutoff
