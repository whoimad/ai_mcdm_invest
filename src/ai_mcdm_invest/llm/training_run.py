"""Training-year run: the real model reads every fiscal 2014–2016 auditor's report.

For each 10-K in the evaluation universe of the training years, the locate
step cuts the auditor's reports (or the whole of Items 8 and 15, a fallback
read) and the extraction reads them. No fiscal 2017 or 2018 filing is read.

Run with `uv run python -m ai_mcdm_invest.llm.training_run [--limit N]`. It
needs Ollama with `qwen2.5:7b` and a GPU. A killed run resumes where it
stopped: finished filings are not read again. `--limit` reads a fixed sample,
a pilot whose answers the full run then reuses.
"""

import argparse
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.corpus import iter_record_sections
from ai_mcdm_invest.data.panel import FISCAL_YEARS, TEST_YEARS
from ai_mcdm_invest.llm.going_concern import (
    CLEAR,
    FAILED,
    FLAGGED,
    LLMClient,
    extract_going_concern,
    filing_status,
    write_whole,
)
from ai_mcdm_invest.llm.keyword_benchmark import KEYWORD_VERSION, keyword_match
from ai_mcdm_invest.llm.locate import Located, locate_auditor_reports
from ai_mcdm_invest.llm.ollama import OllamaClient
from ai_mcdm_invest.tables import write_table

TRAINING_YEARS = [year for year in FISCAL_YEARS if year not in TEST_YEARS]
SAMPLE_SEED = 0
KEY = ["cik", "fiscal_year"]
# Items 8 and 15 hold no audit opinion text at all: nothing is sent to the model,
# and the filing is left out of the evaluation universe (decision 5.2).
NO_REPORT = "no_report"

# (cik, corpus_year) -> the record's Item texts ("8" -> text).
SectionsLoader = Callable[[set[tuple[int, int]]], Iterable[tuple[tuple[int, int], dict[str, str]]]]


def training_run(
    filings: pd.DataFrame,
    criteria: pd.DataFrame,
    load_sections: SectionsLoader,
    client: LLMClient,
    store_dir: Path,
    *,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the going-concern facts of every training-year 10-K in the evaluation
    universe. See `going_concern_run`."""
    return going_concern_run(
        filings, criteria, load_sections, client, store_dir, years=TRAINING_YEARS, limit=limit
    )


def going_concern_run(
    filings: pd.DataFrame,
    criteria: pd.DataFrame,
    load_sections: SectionsLoader,
    client: LLMClient,
    store_dir: Path,
    *,
    years: Iterable[int],
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the going-concern facts of every 10-K of fiscal `years` in the evaluation universe.

    `filings` is the filings table and `criteria` the ratio criteria (for
    `too_few_ratios`). `load_sections` gives the Item texts of the corpus
    records asked for. `limit` reads a fixed random sample of that many filings.

    Returns one row per filing (`cik`, `fiscal_year`, `accession`, `status`,
    `fallback_read`, `chunks`, `seconds`, `keyword_flagged`, `keyword_version`;
    a filing whose Items 8 and 15 hold no audit opinion text has status
    `NO_REPORT`, no chunks, no seconds and `keyword_flagged` false) and one row per
    chunk (the filing's key, then the extraction's result, see
    `extract_going_concern`, then `keyword_match`, the sentence the keyword
    benchmark flags in the same chunk, if any). `seconds` is the time the model
    took on the filing, kept from the run that read it.
    """
    selected = _universe(filings, criteria, years)
    if limit is not None:
        selected = selected.sample(n=min(limit, len(selected)), random_state=SAMPLE_SEED)
    chunks, located = located_chunks(selected, load_sections)

    counting = _CountingClient(client)
    rows = []
    seconds = {}
    for accession, filing_chunks in chunks.groupby("filing_id", sort=False):
        calls = counting.calls
        start = time.perf_counter()
        rows.append(extract_going_concern(filing_chunks, counting, store_dir))
        elapsed = time.perf_counter() - start
        asked = counting.calls > calls
        seconds[accession] = _seconds(store_dir, str(accession), elapsed, asked)

    chunk_results = pd.concat(rows, ignore_index=True) if rows else _no_chunk_results()
    keyword = chunks[["filing_id", "chunk"]].assign(
        keyword_match=chunks["text"].map(keyword_match).astype(object)
    )
    chunk_results = chunk_results.merge(
        keyword, on=["filing_id", "chunk"], how="left", validate="one_to_one"
    )
    accessions = [accession for accession in selected["accession"] if accession in located]
    no_report = pd.Series({accession: located[accession].no_report for accession in accessions})
    status = (
        filing_status(chunk_results).reindex(accessions).astype(object).mask(no_report, NO_REPORT)
    )
    keys = selected.set_index("accession")[KEY]
    filing_results = (
        status.rename_axis("accession")
        .rename("status")
        .to_frame()
        .join(keys)
        .assign(
            fallback_read=lambda table: [located[a].fallback_read for a in table.index],
            chunks=chunk_results.groupby("filing_id").size().reindex(accessions, fill_value=0),
            seconds=pd.Series(seconds, dtype=float).reindex(accessions),
            keyword_flagged=chunk_results["keyword_match"]
            .notna()
            .groupby(chunk_results["filing_id"])
            .any()
            .reindex(accessions, fill_value=False),
            keyword_version=KEYWORD_VERSION,
        )
        .reset_index()[
            [
                *KEY,
                "accession",
                "status",
                "fallback_read",
                "chunks",
                "seconds",
                "keyword_flagged",
                "keyword_version",
            ]
        ]
    )
    chunk_results = (
        chunk_results.rename(columns={"filing_id": "accession"})
        .merge(keys.reset_index(), on="accession", validate="many_to_one")
        .pipe(lambda table: table[[*KEY, *table.columns.drop(KEY)]])
    )
    return filing_results, chunk_results


def run_counts(filing_results: pd.DataFrame) -> pd.DataFrame:
    """Per fiscal year and for all years: filings, flagged, clear, failed and
    "no report" counts, the fallback reads that were read (a "no report" filing
    is not), and the mean seconds per filing read."""

    def counts(table: pd.DataFrame) -> dict:
        return {
            "filings": len(table),
            FLAGGED: int((table["status"] == FLAGGED).sum()),
            CLEAR: int((table["status"] == CLEAR).sum()),
            FAILED: int((table["status"] == FAILED).sum()),
            NO_REPORT: int((table["status"] == NO_REPORT).sum()),
            "fallback_read": int((table["fallback_read"] & (table["status"] != NO_REPORT)).sum()),
            "seconds_per_filing": round(float(table["seconds"].mean()), 2),
        }

    rows = [
        {"fiscal_year": year} | counts(table)
        for year, table in filing_results.groupby("fiscal_year")
    ]
    rows.append({"fiscal_year": "all"} | counts(filing_results))
    return pd.DataFrame(rows)


def _universe(filings: pd.DataFrame, criteria: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    """10-Ks of fiscal `years` with text in the corpus and at least half the ratios
    (the evaluation universe's rule, decision 5.2). "No report" 10-Ks are kept
    here so the run counts them, but are not read."""
    table = filings.merge(criteria[[*KEY, "too_few_ratios"]], on=KEY, validate="one_to_one")
    kept = (
        table["fiscal_year"].isin(list(years))
        & table["in_corpus"].astype(bool)
        & ~table["too_few_ratios"].astype(bool)
    )
    return table[kept].reset_index(drop=True)


def located_chunks(
    selected: pd.DataFrame, load_sections: SectionsLoader
) -> tuple[pd.DataFrame, dict[str, Located]]:
    """The chunks the model reads, in the order selected, and what was located
    in each selected filing. A "no report" filing has no chunks to read."""
    records = selected.to_dict("records")
    accessions = {(int(row["cik"]), int(row["corpus_year"])): row["accession"] for row in records}
    located = {}
    for key, sections in load_sections(set(accessions)):
        located[accessions[key]] = locate_auditor_reports(sections)
    rows = [
        {
            "filing_id": row["accession"],
            "period_end": pd.Timestamp(row["period_end"]).date().isoformat(),
            "chunk": number,
            "item": chunk.item,
            "text": chunk.text,
        }
        for row in records
        if row["accession"] in located and not located[row["accession"]].no_report
        for number, chunk in enumerate(located[row["accession"]].chunks)
    ]
    return pd.DataFrame(rows, columns=["filing_id", "period_end", "chunk", "item", "text"]), located


def _no_chunk_results() -> pd.DataFrame:
    """The chunk results when every selected filing is "no report"."""
    return pd.DataFrame(columns=["filing_id", "chunk", "status"])


def _seconds(store_dir: Path, accession: str, elapsed: float, asked: bool) -> float:
    """The model's seconds on a filing: `elapsed` if the model was `asked`
    anything for it now, else kept from the run that read it."""
    path = store_dir / "seconds" / f"{accession}.json"
    if asked or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_whole(path, json.dumps(elapsed))
        return elapsed
    return json.loads(path.read_text())


class _CountingClient:
    """The client, counting the prompts it answers."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.calls = 0

    @property
    def model_version(self) -> str:
        return self.client.model_version

    def complete(self, prompt: str, schema: dict) -> str:
        reply = self.client.complete(prompt, schema)
        self.calls += 1
        return reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="read a fixed sample of this many filings")
    limit = parser.parse_args().limit
    settings = Settings.from_env()
    client = OllamaClient()
    print(f"model: {client.model_version}")
    filings, chunks = training_run(
        pd.read_parquet(settings.processed_dir / "filings.parquet"),
        pd.read_parquet(settings.processed_dir / "ratio_criteria.parquet"),
        lambda keys: iter_record_sections(settings.raw_corpus_dir, keys),
        client,
        settings.interim_dir / "going_concern_store",
        limit=limit,
    )
    name = "going_concern_training" + ("" if limit is None else f"_pilot{limit}")
    write_table(filings, settings.processed_dir, f"{name}_filings")
    path = write_table(chunks, settings.processed_dir, f"{name}_chunks")
    counts = run_counts(filings)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report = settings.reports_dir / f"{name}.md"
    report.write_text(
        f"# Going-concern run, fiscal {TRAINING_YEARS[0]}-{TRAINING_YEARS[-1]}\n\n"
        f"Model: `{client.model_version}`\n\n```\n{counts.to_string(index=False)}\n```\n"
    )
    print(f"results: written to {path} and beside it; counts in {report}")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
