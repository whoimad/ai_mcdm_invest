"""Company-link stage: each panel row linked to its SEC filer (CIK).

The panel knows companies only by ticker, and tickers from 2014-2018 have since
been renamed, delisted or reused. So each row is linked by its own figures:
its total assets, plus its revenue or net income, must equal a filer's XBRL
figures for a period ending in the same calendar year as the row's fiscal
year, within `TOLERANCE`. (The panel names a fiscal year by the year its period
ends: a retailer's year ending January 2017 is fiscal 2017. 99% of real matches
end in that year.) A match to exactly one filer counts; a match to several is
rejected as ambiguous. Today's ticker list is used only when the figures find
no filer, and where the two disagree the figure match wins. After an
ambiguous match the ticker link is kept only if its filer is one of those the
figures matched.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_mcdm_invest.data.panel import KEY_COLUMNS
from ai_mcdm_invest.data.sec_download import ASSETS_TAG, NET_INCOME_TAG, REVENUE_TAGS

# Panel figures equal the XBRL figures to the dollar for a true match; a looser
# tolerance only adds false candidates (0.1% was the best of 0.1-5% in the
# fiscal 2017 prototype, at about 99% precision).
TOLERANCE = 0.001

FIGURE_OF_TAG = {ASSETS_TAG: "total_assets", NET_INCOME_TAG: "net_income"} | {
    tag: "revenue" for tag in REVENUE_TAGS
}
PANEL_COLUMN = {"total_assets": "Total assets", "revenue": "Revenue", "net_income": "Net Income"}

HIT_COLUMNS = [
    *KEY_COLUMNS,
    "cik",
    "filer_name",
    "period_end",
    "matched_revenue",
    "matched_net_income",
]
LINK_COLUMNS = [
    *KEY_COLUMNS,
    "cik",
    "link_method",
    "figure_match",
    "figure_cik",
    "ticker_cik",
    "filer_name",
    "period_end",
    "matched_revenue",
    "matched_net_income",
]


def load_xbrl_figures(frames_dir: Path) -> pd.DataFrame:
    """Every frame under `frames_dir` as one long table.

    Columns: `cik`, `filer_name`, `period_end` (ISO date), `figure`
    (`total_assets`, `revenue` or `net_income`) and `value` in USD.
    """
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"XBRL frames not found: {frames_dir}. Run `make fetch` first.")
    records = []
    for path in sorted(frames_dir.glob("*.json")):
        figure = FIGURE_OF_TAG[path.stem.split("_CY")[0]]
        records += [
            (point["cik"], point["entityName"], point["end"], figure, point["val"])
            for point in json.loads(path.read_text())["data"]
        ]
    figures = pd.DataFrame.from_records(
        records, columns=["cik", "filer_name", "period_end", "figure", "value"]
    )
    return figures.astype({"value": float}).drop_duplicates(ignore_index=True)


def load_ticker_ciks(path: Path) -> pd.DataFrame:
    """Today's SEC ticker list as a table of `ticker` and `cik`, one row per ticker."""
    if not path.is_file():
        raise FileNotFoundError(f"SEC ticker list not found: {path}. Run `make fetch` first.")
    entries = json.loads(path.read_text()).values()
    tickers = pd.DataFrame(
        {"ticker": [e["ticker"].upper() for e in entries], "cik": [e["cik_str"] for e in entries]}
    )
    return tickers.drop_duplicates("ticker", ignore_index=True)


def link_companies(
    panel: pd.DataFrame, xbrl_figures: pd.DataFrame, ticker_ciks: pd.DataFrame
) -> pd.DataFrame:
    """One row per panel row (`ticker`, `fiscal_year`) with its filer link.

    - `cik`: the filer the row is linked to, empty if unlinked
    - `link_method`: `figure_match`, `ticker` or `unlinked`
    - `figure_match`: `matched` (one filer), `ambiguous` (several, rejected)
      or `none`
    - `figure_cik`, `ticker_cik`: what each method found on its own
    - `filer_name`, `period_end`, `matched_revenue`, `matched_net_income`:
      the evidence of a figure match
    """
    hits = _figure_hits(panel, xbrl_figures)
    filers_per_row = pd.Series(hits.groupby(KEY_COLUMNS)["cik"].nunique(), name="n_filers")
    unique_hits = hits.join(filers_per_row, on=KEY_COLUMNS).query("n_filers == 1")
    unique_hits = unique_hits.drop_duplicates(KEY_COLUMNS)

    links = panel[KEY_COLUMNS].merge(filers_per_row.reset_index(), how="left", on=KEY_COLUMNS)
    links = links.merge(
        unique_hits.drop(columns="n_filers").rename(columns={"cik": "figure_cik"}),
        how="left",
        on=KEY_COLUMNS,
    )
    links = links.merge(ticker_ciks.rename(columns={"cik": "ticker_cik"}), how="left", on="ticker")
    # Ticker CIKs the figures confirmed, though they also matched other filers.
    confirmed = hits[["ticker", "fiscal_year", "cik"]].rename(columns={"cik": "ticker_cik"})
    confirmed = confirmed.drop_duplicates().assign(ticker_confirmed=True)
    links = links.merge(confirmed, how="left", on=[*KEY_COLUMNS, "ticker_cik"])

    n_filers = links["n_filers"].fillna(0)
    links["figure_match"] = np.select(
        [n_filers == 1, n_filers > 1], ["matched", "ambiguous"], default="none"
    )
    links["figure_cik"] = links["figure_cik"].astype("Int64")
    links["ticker_cik"] = links["ticker_cik"].astype("Int64")
    ticker_usable = links["ticker_cik"].notna() & (
        (links["figure_match"] == "none") | links["ticker_confirmed"].eq(True)
    )
    links["cik"] = links["figure_cik"].fillna(links["ticker_cik"].where(ticker_usable))
    links["link_method"] = np.select(
        [links["figure_cik"].notna(), ticker_usable],
        ["figure_match", "ticker"],
        default="unlinked",
    )
    for column in ["matched_revenue", "matched_net_income"]:
        links[column] = links[column].fillna(False).astype(bool)
    return links[LINK_COLUMNS].sort_values(KEY_COLUMNS, ignore_index=True)


def _figure_hits(panel: pd.DataFrame, xbrl_figures: pd.DataFrame) -> pd.DataFrame:
    """Every (panel row, filer, period end) whose figures match.

    Total assets must match, and revenue or net income too. Only periods
    ending in the row's own fiscal year are candidates.
    """
    xbrl = xbrl_figures.assign(period_year=xbrl_figures["period_end"].str[:4].astype(int))
    assets = xbrl[xbrl["figure"] == "total_assets"].sort_values("value", ignore_index=True)

    rows = panel[[*KEY_COLUMNS, *PANEL_COLUMN.values()]]
    rows = rows[rows["Total assets"] > 0]
    candidates = []
    for year, year_rows in rows.groupby("fiscal_year"):
        year_assets = assets[assets["period_year"] == year]
        values = year_assets["value"].to_numpy()
        target = year_rows["Total assets"].to_numpy()
        low = np.searchsorted(values, target * (1 - TOLERANCE), side="left")
        high = np.searchsorted(values, target * (1 + TOLERANCE), side="right")
        row_index = np.repeat(np.arange(len(year_rows)), high - low)
        asset_index = np.concatenate([np.arange(lo, hi) for lo, hi in zip(low, high, strict=True)])
        matched = year_rows.iloc[row_index].reset_index(drop=True)
        filers = year_assets.iloc[asset_index].reset_index(drop=True)
        candidates.append(pd.concat([matched, filers[["cik", "filer_name", "period_end"]]], axis=1))
    candidates = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    if candidates.empty:
        return pd.DataFrame(columns=HIT_COLUMNS)

    for figure in ["revenue", "net_income"]:
        column = PANEL_COLUMN[figure]
        reported = xbrl.loc[xbrl["figure"] == figure, ["cik", "period_end", "value"]]
        candidate_keys = [*KEY_COLUMNS, "cik", "period_end"]
        pairs = candidates[[*candidate_keys, column]].merge(reported, on=["cik", "period_end"])
        usable = pairs[column] > 0 if figure == "revenue" else pairs[column] != 0
        close = (pairs[column] - pairs["value"]).abs() <= TOLERANCE * pairs["value"].abs()
        found = pairs.loc[usable & close, candidate_keys]
        found = found.drop_duplicates().assign(**{f"matched_{figure}": True})
        candidates = candidates.merge(found, how="left", on=candidate_keys)
        candidates[f"matched_{figure}"] = candidates[f"matched_{figure}"].fillna(False)

    hits = candidates[candidates["matched_revenue"] | candidates["matched_net_income"]]
    return hits[HIT_COLUMNS]
