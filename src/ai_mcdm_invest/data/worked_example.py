"""Worked example stage: quoted 10-K passages of one company, for the exploratory section.

The report shows one company's fundamentals beside what its 10-K says, to
show at a glance what text adds that numbers cannot. The company is chosen by
hand (`Settings.worked_example`), from the development years only: no test-year filing
is read before the pre-registration (decision 5.1).

Three passages are quoted, each the first paragraph of its Item that fits:
- Item 8 or 15, the auditor's report: the going-concern explanatory paragraph
  ("prepared assuming that the Company will continue as a going concern"),
  else any paragraph saying there is substantial doubt about it;
- Item 1A, risk factors: the first paragraph with going-concern wording;
- Item 3, legal proceedings: the first paragraph.
Paragraphs shorter than `MIN_PASSAGE_CHARS` are headings and are skipped.
"""

import re
from pathlib import Path

import pandas as pd

from ai_mcdm_invest.data.corpus import GOING_CONCERN, load_record_sections, section_body
from ai_mcdm_invest.data.panel import TEST_YEARS

MIN_PASSAGE_CHARS = 80
MAX_PASSAGE_CHARS = 600
CUT_MARK = "[…]"

AUDITOR_GOING_CONCERN = re.compile(
    r"prepared\s+assuming\s+that\s+the\s+\w+\s+will\s+continue\s+as\s+a\s+going[\s-]+concern",
    re.IGNORECASE,
)
SUBSTANTIAL_DOUBT = re.compile(r"substantial\s+doubt", re.IGNORECASE)
# A full stop after a lower-case letter, digit or quote ends a sentence; one after a
# capital initial ("Jeffrey D. Benjamin") does not.
SENTENCE_END = re.compile(r"(?<=[a-z0-9)\"”’][.!?])\s+")

PASSAGE_COLUMNS = ["ticker", "cik", "fiscal_year", "item", "passage"]


def worked_example_passages(
    filings: pd.DataFrame, corpus_dir: Path, ticker: str, fiscal_year: int
) -> pd.DataFrame:
    """The quoted passages of `ticker`'s 10-K for `fiscal_year`, one row per passage.

    Columns: `ticker`, `cik`, `fiscal_year`, `item` and `passage`, in the order
    Item 8 or 15, Item 1A, Item 3; an Item with no fitting paragraph is left
    out. Passages longer than `MAX_PASSAGE_CHARS` are cut after a sentence.
    """
    if fiscal_year in TEST_YEARS:
        raise ValueError(
            f"fiscal {fiscal_year} is a test year: the worked example comes from the "
            f"years before fiscal {min(TEST_YEARS)}"
        )
    rows = filings[(filings["ticker"] == ticker) & (filings["fiscal_year"] == fiscal_year)]
    if rows.empty or not rows["in_corpus"].all():
        raise ValueError(f"{ticker} fiscal {fiscal_year} has no 10-K text in the corpus")
    row = rows.iloc[0]
    cik = int(row["cik"])
    sections = load_record_sections(corpus_dir, cik, int(row["corpus_year"]))

    found = [
        _first(sections, ["8", "15"], AUDITOR_GOING_CONCERN)
        or _first(sections, ["8", "15"], SUBSTANTIAL_DOUBT, GOING_CONCERN),
        _first(sections, ["1A"], GOING_CONCERN),
        _first(sections, ["3"]),
    ]
    passages = [
        {
            "ticker": ticker,
            "cik": cik,
            "fiscal_year": fiscal_year,
            "item": item,
            "passage": _cut(text),
        }
        for item, text in filter(None, found)
    ]
    return pd.DataFrame(passages, columns=PASSAGE_COLUMNS)


def _first(
    sections: dict[str, str], items: list[str], *patterns: re.Pattern
) -> tuple[str, str] | None:
    """The first paragraph of `items`, in that order, that matches every pattern."""
    for item in items:
        for paragraph in section_body(sections.get(item, "")).splitlines():
            paragraph = paragraph.strip()
            if len(paragraph) >= MIN_PASSAGE_CHARS and all(p.search(paragraph) for p in patterns):
                return item, paragraph
    return None


def _cut(paragraph: str) -> str:
    """`paragraph`, or its first sentences up to `MAX_PASSAGE_CHARS` followed by `CUT_MARK`."""
    if len(paragraph) <= MAX_PASSAGE_CHARS:
        return paragraph
    kept = ""
    for sentence in SENTENCE_END.split(paragraph):
        if len(kept) + len(sentence) + len(CUT_MARK) + 2 > MAX_PASSAGE_CHARS:
            break
        kept = f"{kept} {sentence}".strip()
    words = paragraph[:MAX_PASSAGE_CHARS].rsplit(" ", 1)[0]  # no sentence end fits
    return f"{kept or words} {CUT_MARK}"
