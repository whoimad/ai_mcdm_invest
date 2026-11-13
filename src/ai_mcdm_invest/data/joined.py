"""Joined-panel stage: one row per SEC filer and fiscal year.

Each clean-panel row carries the filer its company link found. Rows without
a filer are left out. Several rows can share a filer in a year: share classes
of one company (GOOG and GOOGL), its preferred shares and notes, which the
panel lists with the company's own figures, and, rarely, a reused ticker that
points at a filer another row's figures matched. One row is kept per filer
and year, the first of:

1. rows with a market cap (preferred shares and notes have none in the panel);
2. rows linked by their figures (a ticker-only link to a filer that another
   row's figures matched is a reused ticker, such as BBT pointing at Berkshire
   Hills);
3. the largest market cap (the main share class);
4. the shortest ticker, then alphabetical (class suffixes lengthen a ticker),
   so the choice is deterministic when nothing else decides.
"""

import pandas as pd

from ai_mcdm_invest.data.panel import KEY_COLUMNS

JOINED_KEY = ["cik", "fiscal_year"]


def join_panel(clean: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """The joined panel: one row per filer (`cik`) and fiscal year.

    Columns: `cik`, `fiscal_year`, `ticker`, `link_method`, then every
    clean-panel column (`sector`, the fundamentals, `next_year_return`,
    `tail_loss`) in clean-panel order.
    """
    linked = links.loc[links["cik"].notna(), [*KEY_COLUMNS, "cik", "link_method"]]
    rows = clean.merge(linked, on=KEY_COLUMNS, validate="one_to_one")
    market_cap = rows["Market Cap"]
    ranked = rows.assign(
        _has_market_cap=market_cap.notna(),
        _by_figures=rows["link_method"] == "figure_match",
        _market_cap=market_cap.fillna(0.0),
        _ticker_length=rows["ticker"].str.len(),
    ).sort_values(
        [*JOINED_KEY, "_has_market_cap", "_by_figures", "_market_cap", "_ticker_length", "ticker"],
        ascending=[True, True, False, False, False, True, True],
    )
    kept = ranked.drop_duplicates(JOINED_KEY)
    other_columns = [c for c in clean.columns if c not in KEY_COLUMNS]
    joined = kept[[*JOINED_KEY, "ticker", "link_method", *other_columns]]
    return joined.astype({"cik": "int64"}).reset_index(drop=True)
