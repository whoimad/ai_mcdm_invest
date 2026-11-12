"""Regenerate the fixture raw panel: fictional companies with known answers.

Run with `uv run python tests/fixtures/build_fixture.py`. The CSV files it
writes are committed; tests read those files and never import this script.

Design (the "known answers" tests rely on):
- 25 core companies appear in every fiscal year 2014-2018 with a valid return,
  so the five lowest returns of each year are that year's tail losers.
- LOSERS names those five per year. They differ from year to year.
- Fiscal 2015 is a crash year (every return negative) and fiscal 2016 a boom
  year (every return positive), so a label pooled across years would be wrong.
- Three extra rows must be dropped by cleaning: ZULU (2014, +2500%, a data
  error), ZZNR (2016, no return) and ZZWO (2017, -100%).
- ECHO reports no revenue in 2014 (a missing fundamental that must be kept).
"""

from pathlib import Path

RAW_PANEL_DIR = Path(__file__).parent / "raw" / "2014 2018"

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
HEADER = "Revenue,Net Income,Total assets,Total debt,Market Cap,Sector"


def rows(year: int) -> list[str]:
    loser_returns, floor, step = RETURNS[year]
    returns = dict(zip(LOSERS[year], loser_returns, strict=True))
    others = [t for t in CORE if t not in returns]
    returns |= {t: floor + step * i for i, t in enumerate(others)}
    listed = [(t, returns[t]) for t in CORE] + EXTRA.get(year, [])

    out = [f",{HEADER},{year + 1} PRICE VAR [%],Class"]
    for i, (ticker, ret) in enumerate(listed):
        revenue = 1000.0 * (i + 1) + 10 * (year - 2014)
        fundamentals = [revenue, revenue * 0.1, revenue * 2, revenue * 0.5, revenue * 3]
        cells = [f"{v:.1f}" for v in fundamentals]
        if (ticker, year) == ("ECHO", 2014):
            cells[0] = ""
        ret_cell = "" if ret is None else f"{ret}"
        label = "" if ret is None else str(int(ret > 0))
        out.append(",".join([ticker, *cells, SECTORS[i % 5], ret_cell, label]))
    return out


if __name__ == "__main__":
    RAW_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for year in RETURNS:
        path = RAW_PANEL_DIR / f"{year}_Financial_Data.csv"
        path.write_text("\n".join(rows(year)) + "\n")
