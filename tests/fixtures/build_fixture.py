"""Regenerate the fixture raw panel: fictional companies with known answers.

Run with `uv run python tests/fixtures/build_fixture.py`, then re-record the
fixture manifest:

    AI_MCDM_RAW_DIR=tests/fixtures/raw \\
    AI_MCDM_MANIFEST=tests/fixtures/raw_manifest.json make manifest

The files it writes are committed; tests read those files and never import
this script.

Design (the "known answers" tests rely on):
- 25 core companies appear in every fiscal year 2014-2018 with a valid return,
  so the five lowest returns of each year are that year's tail losers.
- LOSERS names those five per year. They differ from year to year.
- Fiscal 2015 is a crash year (every return negative) and fiscal 2016 a boom
  year (every return positive), so a label pooled across years would be wrong.
- Three extra rows must be dropped by cleaning: ZULU (2014, +2500%, a data
  error), ZZNR (2016, no return) and ZZWO (2017, -100%).
- ECHO reports no revenue in 2014 (a missing fundamental that must be kept).
- LIMA has two more securities in fiscal 2018, reporting LIMA's own figures:
  LIMAK, a second share class with twice LIMA's market cap, and LIMAP, a
  preferred share with no market cap. All three match LIMA's filer, and the
  joined panel must keep LIMAK alone. LIMAP's -30% makes it a sixth tail
  loser in 2018 (27 rows), one the deduplication drops.

SEC files (`raw/sec/`), for the company-link stage. Core company number i
files under CIK 1000 + i, and XBRL frames report its exact figures, except:
- ALFA: its ticker now belongs to another filer (CIK 9001), a reused ticker.
  The figure match (CIK 1000) must win.
- BRVO: renamed, so missing from today's ticker list; found by figures alone.
- CHRL: missing from the ticker list, and a second filer (CIK 9002) reports
  the same figures, so the match is ambiguous and CHRL stays unlinked.
- DLTA: neither in the ticker list nor in XBRL: unlinked.
- FXTR: in the ticker list but absent from XBRL: linked by ticker only.
- GOLF: a second filer (CIK 9003) reports the same figures, but GOLF's own
  CIK is among the two, so its ticker link is confirmed and kept.
- HOTL: its ticker now belongs to CIK 9004, and its figures match two other
  filers (its own and CIK 9005), neither of them 9004: unlinked.
- ECHO 2014 has no revenue in the panel, so it matches on net income.
"""

import json
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_PANEL_DIR = RAW_DIR / "2014 2018"
RAW_SEC_DIR = RAW_DIR / "sec"

CORE = [
    "ALFA", "BRVO", "CHRL", "DLTA", "ECHO", "FXTR", "GOLF", "HOTL", "INDG",
    "JULT", "KILO", "LIMA", "MIKE", "NOVM", "OSCR", "PAPA", "QUBC", "ROMO",
    "SIER", "TNGO", "UNFM", "VCTR", "WSKY", "XRAY", "YANK",
]  # fmt: skip
SECTORS = ["Technology", "Energy", "Healthcare", "Utilities", "Industrials"]

LOSERS = {
    2014: ["ALFA", "BRVO", "CHRL", "DLTA", "ECHO"],
    2015: ["FXTR", "GOLF", "HOTL", "INDG", "JULT"],
    2016: ["KILO", "LIMA", "MIKE", "NOVM", "OSCR"],
    2017: ["PAPA", "QUBC", "ROMO", "SIER", "TNGO"],
    2018: ["UNFM", "VCTR", "WSKY", "XRAY", "YANK"],
}
# (returns of the five losers, lowest return among the other twenty, step)
RETURNS = {
    2014: ([-70.0, -62.5, -55.0, -48.0, -41.0], -20.0, 5.0),
    2015: ([-90.0, -85.0, -80.0, -75.0, -70.0], -60.0, 3.0),  # crash: max -3
    2016: ([1.0, 2.0, 3.0, 4.0, 5.0], 10.0, 10.0),  # boom: min +1
    2017: ([-66.0, -59.0, -52.0, -45.0, -38.0], -15.0, 4.0),
    2018: ([-77.0, -64.0, -51.0, -44.0, -37.0], -25.0, 6.0),
}
EXTRA = {
    2014: [("ZULU", 2500.0)],
    2016: [("ZZNR", None)],
    2017: [("ZZWO", -100.0)],
}
# Other securities of a core company: (ticker, issuer, market cap multiple, return).
SHARE_CLASSES = {
    2018: [("LIMAK", "LIMA", 2.0, 40.0), ("LIMAP", "LIMA", None, -30.0)],
}
HEADER = "Revenue,Net Income,Total assets,Total debt,Market Cap,Sector"


NOT_IN_TICKER_LIST = {"BRVO", "CHRL", "DLTA"}
REUSED_TICKERS = {"ALFA": 9001, "HOTL": 9004}
NOT_IN_XBRL = {"DLTA", "FXTR"}
SAME_FIGURES_AS = {"CHRL": 9002, "GOLF": 9003, "HOTL": 9005}


def cik(ticker: str) -> int:
    return 1000 + CORE.index(ticker)


def figures(i: int, year: int) -> tuple[float, float, float, float, float]:
    """Revenue, net income, total assets, total debt and market cap of row i."""
    revenue = 1000.0 * (i + 1) + 10 * (year - 2014)
    return revenue, revenue * 0.1, revenue * 2, revenue * 0.5, revenue * 3


def rows(year: int) -> list[str]:
    loser_returns, floor, step = RETURNS[year]
    returns = dict(zip(LOSERS[year], loser_returns, strict=True))
    others = [t for t in CORE if t not in returns]
    returns |= {t: floor + step * i for i, t in enumerate(others)}
    listed = [(t, returns[t]) for t in CORE] + EXTRA.get(year, [])

    out = [f",{HEADER},{year + 1} PRICE VAR [%],Class"]
    for i, (ticker, ret) in enumerate(listed):
        cells = [f"{v:.1f}" for v in figures(i, year)]
        if (ticker, year) == ("ECHO", 2014):
            cells[0] = ""
        ret_cell = "" if ret is None else f"{ret}"
        label = "" if ret is None else str(int(ret > 0))
        out.append(",".join([ticker, *cells, SECTORS[i % 5], ret_cell, label]))
    for ticker, issuer, cap_multiple, ret in SHARE_CLASSES.get(year, []):
        i = CORE.index(issuer)
        cells = [f"{v:.1f}" for v in figures(i, year)]
        cells[4] = "" if cap_multiple is None else f"{figures(i, year)[4] * cap_multiple:.1f}"
        out.append(",".join([ticker, *cells, SECTORS[i % 5], f"{ret}", str(int(ret > 0))]))
    return out


def ticker_list() -> dict:
    listed = [t for t in CORE if t not in NOT_IN_TICKER_LIST]
    ciks = {t: REUSED_TICKERS.get(t, cik(t)) for t in listed}
    return {
        str(n): {"cik_str": ciks[t], "ticker": t, "title": f"{t} Corp"}
        for n, t in enumerate(listed)
    }


def frames(year: int) -> dict[str, dict]:
    """The year's XBRL frames: total assets, revenue and net income per filer."""
    assets, revenue, net_income = [], [], []
    for i, ticker in enumerate(CORE):
        if ticker in NOT_IN_XBRL:
            continue
        rev, ni, total_assets, _, _ = figures(i, year)
        filers = [cik(ticker)] + ([SAME_FIGURES_AS[ticker]] if ticker in SAME_FIGURES_AS else [])
        for filer in filers:
            point = {"cik": filer, "entityName": f"{ticker} Corp", "end": f"{year}-12-31"}
            assets.append(point | {"val": total_assets})
            revenue.append(point | {"start": f"{year}-01-01", "val": rev})
            net_income.append(point | {"start": f"{year}-01-01", "val": ni})
    return {
        f"Assets_CY{year}Q4I": {"data": assets},
        f"Revenues_CY{year}": {"data": revenue},
        f"NetIncomeLoss_CY{year}": {"data": net_income},
    }


if __name__ == "__main__":
    RAW_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_SEC_DIR / "frames").mkdir(parents=True, exist_ok=True)
    for year in RETURNS:
        path = RAW_PANEL_DIR / f"{year}_Financial_Data.csv"
        path.write_text("\n".join(rows(year)) + "\n")
        for name, frame in frames(year).items():
            (RAW_SEC_DIR / "frames" / f"{name}.json").write_text(json.dumps(frame, indent=1))
    (RAW_SEC_DIR / "company_tickers.json").write_text(json.dumps(ticker_list(), indent=1))
