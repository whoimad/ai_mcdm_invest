"""Locate: narrow a 10-K to the auditor's report, the block the LLM reads.

The report is found by document structure, in Items 8 and 15: its heading
("Report of Independent Registered Public Accounting Firm") down to the
auditor's signature date. Each report found is its own chunk.

A heading counts only when an audit opinion opens within `OPENING_CHARS` of
it ("We have audited", or "In our opinion" in reports that lead with the
opinion), before any other heading: an index or a cross-reference names the
report without one. The report ends at the first line after that opening that
starts with a date (the auditor signs and dates it), else at the next heading,
and is capped at `CHUNK_CHARS` either way. Some reports in the corpus lost
their signature in conversion, so a missing date does not drop the report.

When neither Item holds a report, both are read in full, in chunks of up to
`CHUNK_CHARS` cut between lines, and the 10-K is marked "fallback read".
"""

import re
from dataclasses import dataclass

from ai_mcdm_invest.data.text import CHARS_PER_TOKEN, LONG_DATE_PATTERN

REPORT_ITEMS = ["8", "15"]
CHUNK_TOKENS = 6000
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN

HEADING = re.compile(
    r"report\s+of\s+independent\s+registered\s+public\s+accounting\s+firms?", re.IGNORECASE
)
OPENING_CHARS = 1500
AUDIT_OPENING = re.compile(r"we\s+have\s+(?:audited|examined)|in\s+our\s+opinion", re.IGNORECASE)
# The auditor's signature date: a line holding a date, perhaps after the city
# ("Denver, Colorado, March 14, 2016"), that ends the line or goes on with a
# second date ("March 14, 2016, except for Note 2, as to which the date is ...").
# A date that goes on with other words ("December 31, 2015 and 2014, and the
# related statements") is part of the opinion, wrapped onto a new line.
SIGNATURE_DATE = re.compile(
    rf"^[^\S\n]*(?:[^\n]{{0,80}},[^\S\n]*)?(?:dated:?[^\S\n]*)?{LONG_DATE_PATTERN}"
    r"[^\S\n]*(?:$|[,;(][^\S\n]*except\b.*$)",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Chunk:
    """One block of text the LLM reads, and the Item it comes from."""

    item: str
    text: str


@dataclass(frozen=True)
class Located:
    """The chunks of one 10-K; `fallback_read` when no auditor's report was found."""

    chunks: tuple[Chunk, ...]
    fallback_read: bool


def locate_auditor_reports(sections: dict[str, str]) -> Located:
    """The auditor's reports in Items 8 and 15 of a 10-K (`sections`: Item -> text)."""
    chunks = []
    for item in REPORT_ITEMS:
        text = sections.get(item, "")
        chunks += [Chunk(item, text[start:end]) for start, end in _report_spans(text)]
    if chunks:
        return Located(tuple(chunks), fallback_read=False)
    chunks = [
        Chunk(item, piece)
        for item in REPORT_ITEMS
        for piece in _split_between_lines(sections.get(item, ""))
    ]
    return Located(tuple(chunks), fallback_read=True)


def _report_spans(text: str) -> list[tuple[int, int]]:
    """Start and end of each auditor's report in one Item's text."""
    headings = [match.span() for match in HEADING.finditer(text)]
    starts = [start for start, _ in headings] + [len(text)]
    spans = []
    for (start, heading_end), next_heading in zip(headings, starts[1:], strict=True):
        opening = AUDIT_OPENING.search(
            text, heading_end, min(heading_end + OPENING_CHARS, next_heading)
        )
        if not opening:
            continue
        limit = min(next_heading, start + CHUNK_CHARS)
        signed = SIGNATURE_DATE.search(text, opening.end(), limit)
        spans.append((start, signed.end() if signed else limit))
    return spans


def _split_between_lines(text: str) -> list[str]:
    """`text` in consecutive pieces of up to `CHUNK_CHARS`, cut after a line end if possible."""
    pieces = []
    while len(text) > CHUNK_CHARS:
        cut = text.rfind("\n", 0, CHUNK_CHARS) + 1 or CHUNK_CHARS
        pieces.append(text[:cut])
        text = text[cut:]
    return [*pieces, text] if text else pieces
