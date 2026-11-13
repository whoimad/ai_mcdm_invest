"""EDGAR-CORPUS: 10-Ks split by Item, reduced to facts about each record.

The corpus (Hugging Face `eloukas/edgar-corpus`) holds one JSON line per
10-K with `cik`, a `year` label and one `section_<Item>` text per Item. The
label is not reliable: it is usually the fiscal year covered, but some 10-Ks
are labelled by the year they were filed (AMD's and Amazon's fiscal 2018
10-Ks are labelled 2019) and some by the year a retail fiscal year starts.
So each record's period is read from its own text: the date it mentions most
in Items 7, 8 and 15 (the balance-sheet date and "year ended" dates; on a
tie, the latest). The filings stage matches that date to the report date the
SEC lists for each 10-K.

Raw text stays in the corpus files; only these facts are kept.
"""

import hashlib
import json
import multiprocessing
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Items the filings table reports on: presence, estimated tokens.
PRESENCE_ITEMS = ["1", "1A", "3", "7", "8", "9A", "15"]
TOKEN_ITEMS = ["1", "1A", "7", "8", "15"]
# The facts the filings table copies from a record, with their table dtypes
# (nullable, since rows without corpus text leave them empty).
FACT_DTYPES = (
    {f"has_item_{item}": "boolean" for item in PRESENCE_ITEMS}
    | {f"tokens_item_{item}": "Int64" for item in TOKEN_ITEMS}
    | {"tokens_total": "Int64", "gc_keyword_items": "string"}
)
CHARS_PER_TOKEN = 4

# A section is a stub, and counts as absent, when what is left after its
# headings and titles is shorter than `MIN_BODY_CHARS`, holds no sentence (a
# leftover table of contents), or is shorter than `STUB_MAX_CHARS` and opens
# (within `STUB_OPENING_CHARS`) by saying the Item is omitted, does not apply,
# or is found elsewhere (another Item, a page, a note). Wording further in is
# ignored: a short real Item 9A may say a schedule "is not applicable".
MIN_BODY_CHARS = 40  # drops a leftover title such as "RISK FACTORS"
STUB_MAX_CHARS = 1000
STUB_OPENING_CHARS = 300
STUB_WORDING = re.compile(
    r"not applicable|not required|omitted|smaller reporting compan|by\s+reference"
    r"|reference is made|(?:is|are)\s+set\s+forth\s+(?:in|under)"
    r"|\b(?:see|refer to)\s+(?:the\s+)?(?:item|part|page|note|section)"
    r"|(?:included|contained|set forth|appears?|presented|found|located|listed)\s+"
    r"(?:in|under|on|at|beginning)\s+(?:the\s+)?(?:item|part|page|pages|accompanying|index)"
    r"|begin\w*\s+(?:immediately\s+)?(?:after|following)|\b(?:on|at)\s+pages?\b|page\s+f-\d"
    r"|index to (?:the\s+)?(?:consolidated\s+)?financial"
    r"|^\s*none\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
SENTENCE = re.compile(r"[a-z0-9][.!?]", re.IGNORECASE)
HEADING_LINE = re.compile(r"^\s*(?:part\s+[ivx]+|item\s+\d+[a-z]?)\b.{0,100}$", re.IGNORECASE)
# A title line: short, and every word of four letters or more capitalised
# ("Financial Statements and Supplementary Data.").
TITLE_MAX_CHARS = 150
LONG_WORD = re.compile(r"[A-Za-z][A-Za-z’'-]{3,}")

GOING_CONCERN = re.compile(r"going[\s-]+concern", re.IGNORECASE)

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
LONG_DATE = re.compile(rf"\b({MONTHS})\s+(\d{{1,2}}),\s*((?:19|20)\d\d)\b")
PERIOD_ITEMS = ["7", "8", "15"]

# Item 1A paragraphs shorter than this are headings, not risk factors.
MIN_PARAGRAPH_CHARS = 100
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
NOT_LETTERS = re.compile(r"[^a-z]+")

CIK_PREFIX = re.compile(rb'^\{\s*"filename"\s*:\s*"(\d+)_')


def load_corpus(corpus_dir: Path, ciks: set[int]) -> pd.DataFrame:
    """Facts about every corpus record of the filers in `ciks`.

    Columns: `cik`, `corpus_year` (the record's label), `period_end` (read
    from the text; empty when the text names no date), `has_item_<Item>` for
    `PRESENCE_ITEMS`, `tokens_item_<Item>` for `TOKEN_ITEMS`, `tokens_total`,
    `gc_keyword_items` (Items with going-concern wording, `;`-separated) and
    `item1a_paragraphs` (each Item 1A paragraph as a tuple of sentence
    fingerprints, for year-on-year comparison).
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"EDGAR-CORPUS not found: {corpus_dir}. Run `make fetch` first.")
    paths = sorted(corpus_dir.glob("*.jsonl"))
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("forkserver")) as pool:
        parts = list(pool.map(_describe_file, paths, [ciks] * len(paths)))
    records = [row for part in parts for row in part]
    columns = ["cik", "corpus_year", "period_end", *FACT_DTYPES, "item1a_paragraphs"]
    corpus = pd.DataFrame.from_records(records, columns=columns)
    corpus["period_end"] = pd.to_datetime(corpus["period_end"])
    return corpus.sort_values(["cik", "corpus_year"], ignore_index=True)


def _describe_file(path: Path, ciks: set[int]) -> list[dict]:
    rows = []
    with path.open("rb") as file:
        for line in file:
            prefix = CIK_PREFIX.match(line)
            if prefix and int(prefix.group(1)) not in ciks:
                continue  # skip parsing records of other filers
            record = json.loads(line)
            if int(record["cik"]) in ciks:
                rows.append(describe_record(record))
    return rows


def load_record_sections(corpus_dir: Path, cik: int, corpus_year: int) -> dict[str, str]:
    """The Item texts (`"1A"` -> text) of the corpus record of `cik` labelled `corpus_year`."""
    for path in sorted(corpus_dir.glob(f"{corpus_year}_*.jsonl")):
        with path.open("rb") as file:
            for line in file:
                prefix = CIK_PREFIX.match(line)
                if prefix and int(prefix.group(1)) != cik:
                    continue  # skip parsing records of other filers
                record = json.loads(line)
                if int(record["cik"]) == cik and int(record["year"]) == corpus_year:
                    return _sections(record)
    raise FileNotFoundError(
        f"no corpus record for CIK {cik} labelled {corpus_year} in {corpus_dir}"
    )


def _sections(record: dict) -> dict[str, str]:
    return {
        key.removeprefix("section_"): text or ""
        for key, text in record.items()
        if key.startswith("section_")
    }


def describe_record(record: dict) -> dict:
    """The facts `load_corpus` keeps about one corpus record."""
    sections = _sections(record)
    facts: dict = {
        "cik": int(record["cik"]),
        "corpus_year": int(record["year"]),
        "period_end": _period_end(sections),
    }
    for item in PRESENCE_ITEMS:
        facts[f"has_item_{item}"] = not _is_stub(sections.get(item, ""))
    for item in TOKEN_ITEMS:
        facts[f"tokens_item_{item}"] = len(sections.get(item, "")) // CHARS_PER_TOKEN
    facts["tokens_total"] = sum(len(text) for text in sections.values()) // CHARS_PER_TOKEN
    facts["gc_keyword_items"] = ";".join(
        item for item, text in sections.items() if GOING_CONCERN.search(text)
    )
    item_1a = sections.get("1A", "")
    facts["item1a_paragraphs"] = (
        tuple(_paragraph_fingerprints(item_1a)) if facts["has_item_1A"] else ()
    )
    return facts


def _period_end(sections: dict[str, str]) -> str | None:
    """The date the text mentions most in Items 7, 8 and 15, else in any Item."""
    for texts in ([sections.get(item, "") for item in PERIOD_ITEMS], sections.values()):
        counts = Counter(LONG_DATE.findall(" ".join(texts)))
        dates = [(n, day) for (m, d, y), n in counts.items() if (day := _date(m, d, y))]
        if dates:
            return max(dates)[1].isoformat()
    return None


def _date(month: str, day: str, year: str) -> date | None:
    """The date, or None for an impossible one ("June 31") the text happens to contain."""
    try:
        return datetime.strptime(f"{month} {day} {year}", "%B %d %Y").date()
    except ValueError:
        return None


def section_body(section: str) -> str:
    """The section without its heading and title lines ("Item 1A. Risk Factors", "PART II")."""
    lines = [
        line
        for line in section.splitlines()
        if not HEADING_LINE.match(line) and not _is_title(line)
    ]
    return "\n".join(lines).strip()


def _is_title(line: str) -> bool:
    words = LONG_WORD.findall(line)
    return len(line) <= TITLE_MAX_CHARS and bool(words) and all(w[0].isupper() for w in words)


def _is_stub(section: str) -> bool:
    body = section_body(section)
    if len(body) < MIN_BODY_CHARS or not SENTENCE.search(body):
        return True
    opening = body[:STUB_OPENING_CHARS]
    return len(body) < STUB_MAX_CHARS and STUB_WORDING.search(opening) is not None


def _paragraph_fingerprints(section: str) -> list[tuple[bytes, ...]]:
    """Each paragraph of Item 1A as a tuple of fingerprints of its sentences.

    Sentences are compared on their letters only, lower-cased, so changes in
    numbering, punctuation or spacing do not count as changes.
    """
    paragraphs = []
    for paragraph in section_body(section).splitlines():
        if len(paragraph.strip()) < MIN_PARAGRAPH_CHARS:
            continue
        sentences = [
            " ".join(NOT_LETTERS.sub(" ", sentence.lower()).split())
            for sentence in SENTENCE_END.split(paragraph)
        ]
        paragraphs.append(
            tuple(hashlib.blake2b(s.encode(), digest_size=8).digest() for s in sentences if s)
        )
    return [p for p in paragraphs if p]
