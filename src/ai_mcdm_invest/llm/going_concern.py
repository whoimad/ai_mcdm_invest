"""Going-concern extraction: read each located chunk, verify the quote, score in code.

A plain loop over small steps, never an agent: for every chunk of every filing,
the LLM returns the evidence quote first and then yes/no facts about it; code
checks the quote is in the chunk and turns the facts into the score.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

PROMPT_VERSION = "gc-1"
# A failed check is retried twice, then recorded as failed (decision 6.5).
ATTEMPTS = 3

# A chunk's or a filing's status.
FLAGGED = "flagged"
CLEAR = "clear"
FAILED = "extraction_failed"

FIELDS = ["auditor_opinion", "this_company", "current_year", "doubt_expressed", "alleviated"]
# The quote comes first: with yes/no fields first, a pre-test model ruled out
# genuine opinions before reading them.
SCHEMA = {
    "type": "object",
    "properties": {"quote": {"type": "string"}} | {field: {"type": "boolean"} for field in FIELDS},
    "required": ["quote", *FIELDS],
}

PROMPT = """\
You read part of a company's annual report (10-K) for the fiscal year ended {period_end}.

Find the passage that says whether there is substantial doubt about an entity's
ability to continue as a going concern. Copy it word for word into "quote". If
there are several, prefer the independent auditor's own opinion. If there is
none, return an empty quote and false for every other field.

Then answer about the quoted passage only:
- auditor_opinion: it is the independent auditor's report, not management's notes
- this_company: the doubt concerns the company filing this report, not another entity
- current_year: it concerns the fiscal year ended {period_end}, not an earlier year
- doubt_expressed: it states that substantial doubt exists
- alleviated: it states that management's plans alleviate the doubt

Text:
<<<
{text}
>>>
"""
# Run at temperature 0, the same prompt gives the same answer, so a retry says
# what was wrong with the last one.
RETRY_NOTE = """
Your previous answer was rejected: {problem}. This is attempt {attempt} of {attempts}.
Answer again.
"""


class LLMClient(Protocol):
    """The one injected dependency: a model that answers a prompt with JSON."""

    @property
    def model_version(self) -> str: ...

    def complete(self, prompt: str, schema: dict) -> str: ...


def quote_found(quote: str, chunk: str) -> bool:
    """Whether `quote` is in `chunk` word for word, ignoring differences in whitespace."""
    return _squash(quote) in _squash(chunk)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def going_concern_flag(
    *,
    auditor_opinion: bool,
    this_company: bool,
    current_year: bool,
    doubt_expressed: bool,
    alleviated: bool,
) -> bool:
    """The score: flagged only when the auditor's opinion expresses doubt about
    this company for the current fiscal year (decision 6.1).

    `alleviated` is stored for error analysis and does not change the score.
    """
    del alleviated
    return auditor_opinion and this_company and current_year and doubt_expressed


def extract_going_concern(chunks: pd.DataFrame, client: LLMClient, store_dir: Path) -> pd.DataFrame:
    """Read, verify and score every chunk, one filing at a time.

    `chunks` has one row per located chunk (see `locate.py`): `filing_id`,
    `period_end` (the fiscal year end the 10-K reports on), `chunk`, `item` and
    `text`. The result has one row per chunk: `status` (`FLAGGED`, `CLEAR` or
    `FAILED`), the `quote` and the facts (empty when the extraction
    failed), `model_version` and `prompt_version`.

    Under `store_dir`, `responses/` caches every reply and `filings/` holds a
    completion marker per finished filing, so a killed run picks up where it
    stopped. Both are keyed by prompt version, model version and the text read,
    so changing any of them reads the filing again.
    """
    rows = []
    for filing_id, filing in chunks.groupby("filing_id", sort=False):
        # Keyed by the chunks too, so a changed locate rule reads the filing again.
        texts = filing[["period_end", "item", "text"]].astype(str).to_numpy().tolist()
        filing_key = _key(client.model_version, json.dumps([str(filing_id), texts]))
        marker = store_dir / "filings" / f"{filing_key}.json"
        if marker.exists():
            rows += json.loads(marker.read_text())
            continue
        filing_rows = [
            _extract_chunk(chunk, client, store_dir / "responses") for chunk in filing.itertuples()
        ]
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(filing_rows))
        rows += filing_rows
    return pd.DataFrame(rows)


def _extract_chunk(chunk: Any, client: LLMClient, cache_dir: Path) -> dict:
    """One chunk's result row."""
    prompt = PROMPT.format(period_end=chunk.period_end, text=chunk.text)
    answer = _read_chunk(client, prompt, chunk.text, cache_dir)
    if answer is None:
        status = FAILED
        answer = {"quote": None} | dict.fromkeys(FIELDS)
    else:
        flagged = going_concern_flag(**{field: answer[field] for field in FIELDS})
        status = FLAGGED if flagged else CLEAR
    return {
        "filing_id": chunk.filing_id,
        "chunk": chunk.chunk,
        "item": chunk.item,
        "status": status,
        **answer,
        "model_version": client.model_version,
        "prompt_version": PROMPT_VERSION,
    }


def _key(model_version: str, text: str) -> str:
    """A file name for `text` under this prompt version and model version."""
    key = json.dumps([PROMPT_VERSION, model_version, text])
    return hashlib.sha256(key.encode()).hexdigest()


def _read_chunk(client: LLMClient, prompt: str, text: str, cache_dir: Path) -> dict | None:
    """The model's verified answer for one chunk, or None when all `ATTEMPTS` fail."""
    problem = None
    for attempt in range(1, ATTEMPTS + 1):
        asked = prompt
        if problem is not None:
            asked += RETRY_NOTE.format(problem=problem, attempt=attempt, attempts=ATTEMPTS)
        answer, problem = _parse(_complete(client, asked, cache_dir), text)
        if answer is not None:
            return answer
    return None


def _complete(client: LLMClient, prompt: str, cache_dir: Path) -> str:
    """The client's reply to `prompt`, from the cache when this prompt version and
    model version have answered it before (decision 6.5). The prompt holds the
    chunk text and the period end, so a reply is never reused for another chunk."""
    path = cache_dir / f"{_key(client.model_version, prompt)}.json"
    if path.exists():
        return json.loads(path.read_text())["reply"]
    reply = client.complete(prompt, SCHEMA)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_version": client.model_version, "reply": reply}))
    return reply


def _parse(reply: str, text: str) -> tuple[dict | None, str | None]:
    """The answer in `reply` if it passes every check for the chunk `text`, else
    None and what was wrong with it."""
    try:
        answer = json.loads(reply)
    except json.JSONDecodeError:
        return None, "it was not valid JSON"
    if not isinstance(answer, dict) or not isinstance(answer.get("quote"), str):
        return None, 'it had no "quote" text'
    if not all(isinstance(answer.get(field), bool) for field in FIELDS):
        return None, f"it did not give true or false for each of {', '.join(FIELDS)}"
    if not _squash(answer["quote"]) and any(answer[field] for field in FIELDS):
        return None, "it answered true without quoting a passage"
    if not quote_found(answer["quote"], text):
        return None, "the quote is not word for word in the text"
    return {field: answer[field] for field in ["quote", *FIELDS]}, None


def filing_status(chunk_results: pd.DataFrame) -> pd.Series:
    """Each filing's status, indexed by `filing_id`, from its chunks' statuses.

    `FLAGGED` when any chunk is (a verified quote is evidence whatever the other
    chunks say), else `FAILED` when any chunk failed (a failure is never read
    as clear), else `CLEAR`.
    """
    statuses = chunk_results.groupby("filing_id", sort=False)["status"].agg(set)
    return pd.Series(
        {
            filing_id: next(
                (status for status in [FLAGGED, FAILED] if status in found),
                CLEAR,
            )
            for filing_id, found in statuses.items()
        },
        name="status",
    )
