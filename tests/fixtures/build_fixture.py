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
- MIKE's fiscal year ends on 31 January: its XBRL figures end then.

SEC submissions (`raw/sec/submissions.zip`, the bulk file's layout) list a
10-K for every core filer and each period ending 2013-2018, filed on 2 March
of the next year, plus a 10-Q each year. Exceptions:
- ALFA also filed a 10-K/A for fiscal 2016 on 2017-06-01, to be ignored.
- MIKE files on 15 April, and its filings before 2016 sit in an older page
  (`CIK0000001012-submissions-001.json`) as in the real bulk file.
- OSCR filed a late-filing notice (NT 10-K) for fiscal 2016 on 2017-03-16 and
  the 10-K on 2017-04-14.
- QUBC filed no 10-K for fiscal 2017.

EDGAR-CORPUS (`raw/edgar_corpus/<label>_train.jsonl`) holds one record per
10-K above, labelled with the year its period ends, except:
- MIKE is labelled a year earlier (its year ending January 2017 is labelled
  2016, as the real corpus does for some retailers), so picking by label
  would take the wrong filing.
- PAPA's fiscal 2015 10-K is missing, so fiscal 2016 has no prior Item 1A.
- NOVM 2016: Item 1A reads "Not applicable." and Item 8 points to Item 15.
- ROMO 2015: going-concern wording in Items 1A and 8.
- KILO 2016: one of its four Item 1A paragraphs is new and one is reworded
  in one sentence of three; KILO keeps that text afterwards.
Every Item 1 is exactly 2,000 characters (500 estimated tokens).
"""

import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_PANEL_DIR = RAW_DIR / "2014 2018"
RAW_SEC_DIR = RAW_DIR / "sec"
RAW_CORPUS_DIR = RAW_DIR / "edgar_corpus"

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
HEADER = (
    "Revenue,Net Income,Total assets,Total debt,Market Cap,debtEquityRatio,"
    "currentRatio,returnOnEquity,netProfitMargin,priceEarningsRatio,priceToBookRatio,"
    "interestCoverage,Sector"
)


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


def debt_to_equity(i: int) -> str:
    """Debt-to-equity of row i: spread over 0-2, negative (negative equity) for ALFA."""
    return f"{(i * 7) % 11 / 5 - 0.2:.2f}"


def ratios(i: int) -> list[str]:
    """The other five textbook ratios of row i, plus price-to-book, after debt-to-equity.

    ROMO (i = 17) has a negative price-to-earnings ratio; FXTR (i = 5) reports
    no interest coverage.
    """
    current = 0.5 + (i * 3) % 7 / 4
    roe = (i * 5) % 9 / 30 - 0.05
    margin = (i * 4) % 7 / 25
    pe = -8.0 if i == 17 else 5 + (i * 11) % 13 * 3
    pb = 0.4 + (i * 2) % 9 / 2
    coverage = "" if i == 5 else f"{1 + (i * 6) % 10 * 2:.1f}"
    return [f"{current:.2f}", f"{roe:.3f}", f"{margin:.3f}", f"{pe:.1f}", f"{pb:.2f}", coverage]


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
        out.append(
            ",".join(
                [ticker, *cells, debt_to_equity(i), *ratios(i), SECTORS[i % 5], ret_cell, label]
            )
        )
    for ticker, issuer, cap_multiple, ret in SHARE_CLASSES.get(year, []):
        i = CORE.index(issuer)
        cells = [f"{v:.1f}" for v in figures(i, year)]
        cells[4] = "" if cap_multiple is None else f"{figures(i, year)[4] * cap_multiple:.1f}"
        out.append(
            ",".join(
                [
                    ticker,
                    *cells,
                    debt_to_equity(i),
                    *ratios(i),
                    SECTORS[i % 5],
                    f"{ret}",
                    str(int(ret > 0)),
                ]
            )
        )
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
            end = period_end(ticker, year)
            start = (end.replace(year=year - 1) + timedelta(days=1)).isoformat()
            point = {"cik": filer, "entityName": f"{ticker} Corp", "end": end.isoformat()}
            assets.append(point | {"val": total_assets})
            revenue.append(point | {"start": start, "val": rev})
            net_income.append(point | {"start": start, "val": ni})
    return {
        f"Assets_CY{year}Q4I": {"data": assets},
        f"Revenues_CY{year}": {"data": revenue},
        f"NetIncomeLoss_CY{year}": {"data": net_income},
    }


PERIOD_YEARS = range(2013, 2019)
JANUARY_YEAR_END = {"MIKE"}
NO_10K = {("QUBC", 2017)}
NOT_IN_CORPUS = {("PAPA", 2015)}
LABELLED_YEAR_EARLIER = {"MIKE"}
ZIP_DATE = (2026, 11, 12, 0, 0, 0)


def period_end(ticker: str, year: int) -> date:
    """End of the fiscal year `year` (named, as the panel does, by the year it ends in)."""
    return date(year, 1, 31) if ticker in JANUARY_YEAR_END else date(year, 12, 31)


def filing_date(ticker: str, year: int) -> date:
    if ticker in JANUARY_YEAR_END:
        return date(year, 4, 15)
    if (ticker, year) == ("OSCR", 2016):
        return date(2017, 4, 14)
    return date(year + 1, 3, 2)


def submissions_filings(ticker: str) -> list[tuple[str, date, date]]:
    """(form, filing date, report date) of every filing, newest first."""
    filings = []
    for year in PERIOD_YEARS:
        end = period_end(ticker, year)
        if (ticker, year) not in NO_10K:
            filings.append(("10-K", filing_date(ticker, year), end))
        filings.append(("10-Q", end - timedelta(days=50), end - timedelta(days=92)))
    if ticker == "ALFA":
        filings.append(("10-K/A", date(2017, 6, 1), date(2016, 12, 31)))
    if ticker == "OSCR":
        filings.append(("NT 10-K", date(2017, 3, 16), date(2016, 12, 31)))
    return sorted(filings, key=lambda f: f[1], reverse=True)


def columns(filings: list[tuple[str, date, date]], cik_number: int) -> dict:
    """Filings in the bulk file's column layout."""
    return {
        "accessionNumber": [
            f"0000{cik_number}-{d.year % 100:02d}-{n:06d}" for n, (_, d, _) in enumerate(filings)
        ],
        "filingDate": [d.isoformat() for _, d, _ in filings],
        "reportDate": [r.isoformat() for _, _, r in filings],
        "form": [form for form, _, _ in filings],
    }


def write_submissions(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for ticker in CORE:
            number = cik(ticker)
            filings = submissions_filings(ticker)
            recent, older = filings, []
            if ticker == "MIKE":
                recent = [f for f in filings if f[1].year >= 2016]
                older = [f for f in filings if f[1].year < 2016]
            page = f"CIK{number:010d}-submissions-001.json"
            body = {
                "cik": f"{number:010d}",
                "name": f"{ticker} Corp",
                "filings": {
                    "recent": columns(recent, number),
                    "files": [{"name": page, "filingCount": len(older)}] if older else [],
                },
            }
            entries = [(f"CIK{number:010d}.json", body)]
            if older:
                entries.append((page, columns(older, number)))
            for name, content in entries:
                info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
                archive.writestr(info, json.dumps(content), zipfile.ZIP_DEFLATED)


def long_date(day: date) -> str:
    return f"{day:%B} {day.day}, {day.year}"


RISKS = [
    "Our customers may reduce orders. Demand for our products is cyclical. "
    "A downturn in the economy could reduce our revenue.",
    "We depend on a few suppliers. A supplier failure could delay shipments. "
    "Replacing a supplier would take many months.",
    "Our debt limits our flexibility. We must meet covenants under our credit agreement. "
    "A breach could make the debt due at once.",
    "Regulation could raise our costs. New environmental rules may apply to our plants. "
    "Compliance may require large capital spending.",
]
KILO_RISKS = [
    RISKS[0],
    RISKS[1].replace(
        "Replacing a supplier would take many months.", "A new supplier would need a year."
    ),
    "A cyberattack could disrupt our systems. Attackers have targeted firms like ours. "
    "Recovery could be slow and costly.",
    RISKS[3],
]
GOING_CONCERN_RISK = (
    "We have lost money for three years. We may need more capital. "
    "If we cannot raise it, we may not be able to continue as a going concern."
)


def corpus_record(ticker: str, year: int) -> dict:
    end = period_end(ticker, year)
    now, before = long_date(end), long_date(end.replace(year=end.year - 1))
    risks = KILO_RISKS if ticker == "KILO" and year >= 2016 else list(RISKS)
    if (ticker, year) == ("ROMO", 2015):
        risks = [*risks, GOING_CONCERN_RISK]
    item_8 = (
        f"Consolidated balance sheets as of {now} and {before}. "
        f"Statements of operations for the years ended {now} and {before}. "
        f"Statement of cash flows for the year ended {now}."
    )
    if (ticker, year) == ("ROMO", 2015):
        item_8 += (
            " These conditions raise substantial doubt about the Company's ability "
            "to continue as a going concern."
        )
    item_1a = "\n".join(risks)
    if (ticker, year) == ("NOVM", 2016):
        item_1a = "Not applicable."
        item_8 = (
            "The financial statements required by this item are included in Item 15 "
            "of this report, beginning on page F-1."
        )
    item_1 = "Item 1. Business\n" + f"{ticker} Corp makes and sells industrial goods. " * 60
    sections = {
        "section_1": item_1[:2000],
        "section_1A": f"Item 1A. Risk Factors\n{item_1a}\nItem 1B.",
        "section_3": (
            "Item 3. Legal Proceedings\nThe company is party to routine litigation "
            "incidental to its business, none of which is material.\nItem 4."
        ),
        "section_7": (
            "Item 7. Management's Discussion and Analysis\nThis discussion covers the fiscal "
            f"year ended {now} compared with the fiscal year ended {before}. Revenue grew "
            f"during the year ended {now}.\nItem 7A."
        ),
        "section_8": f"Item 8. Financial Statements\n{item_8}\nItem 9.",
        "section_9A": (
            "Item 9A. Controls and Procedures\nManagement evaluated our disclosure controls "
            f"and found them effective as of {now}.\nItem 9B."
        ),
        "section_15": "Item 15. Exhibits\nThe exhibits listed in the index are filed with it.",
    }
    names = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B"]
    names += ["10", "11", "12", "13", "14", "15"]
    label = year - 1 if ticker in LABELLED_YEAR_EARLIER else year
    record = {"filename": f"{cik(ticker)}_{label}.htm", "cik": str(cik(ticker)), "year": str(label)}
    return record | {f"section_{n}": sections.get(f"section_{n}", "") for n in names}


def corpus_files() -> dict[str, list[str]]:
    """JSON lines per corpus file, `<label>_train.jsonl`."""
    files: dict[str, list[str]] = {}
    for ticker in CORE:
        for year in PERIOD_YEARS:
            if (ticker, year) in NO_10K or (ticker, year) in NOT_IN_CORPUS:
                continue
            record = corpus_record(ticker, year)
            files.setdefault(f"{record['year']}_train.jsonl", []).append(
                json.dumps(record, separators=(",", ":"))  # the real files are compact
            )
    return files


if __name__ == "__main__":
    RAW_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_SEC_DIR / "frames").mkdir(parents=True, exist_ok=True)
    for year in RETURNS:
        path = RAW_PANEL_DIR / f"{year}_Financial_Data.csv"
        path.write_text("\n".join(rows(year)) + "\n")
        for name, frame in frames(year).items():
            (RAW_SEC_DIR / "frames" / f"{name}.json").write_text(json.dumps(frame, indent=1))
    (RAW_SEC_DIR / "company_tickers.json").write_text(json.dumps(ticker_list(), indent=1))
    write_submissions(RAW_SEC_DIR / "submissions.zip")
    RAW_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, lines in corpus_files().items():
        (RAW_CORPUS_DIR / name).write_text("\n".join(lines) + "\n")
