"""Training-year run: filings in, one going-concern result per filing out.

The model is a fake and the 10-K text is held in memory, so no GPU or corpus
is needed. The real run is `python -m ai_mcdm_invest.llm.training_run`.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.llm.going_concern import FIELDS
from ai_mcdm_invest.llm.training_run import run_counts, training_run

DOUBT = (
    "The Company has suffered recurring losses from operations that raise substantial "
    "doubt about its ability to continue as a going concern."
)


def report(company: str, body: str = "The statements present fairly.") -> str:
    return (
        "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM\n"
        f"We have audited the balance sheets of {company}.\n{body}\n"
        "/s/ Smith & Jones LLP\nMarch 14, 2016\n"
    )


# 10-K text by (cik, corpus_year): Item -> text.
SECTIONS = {
    (1, 2015): {"8": report("Alfa", DOUBT)},
    (2, 2015): {"8": report("Bravo")},
    (2, 2016): {"8": "Financial statements with no auditor's report.", "15": "Exhibits."},
    (3, 2015): {"8": report("Charlie")},
    (4, 2017): {"8": report("Delta")},
}


def filings_table() -> pd.DataFrame:
    rows = [
        # cik, fiscal_year, in_corpus
        (1, 2015, True),
        (2, 2015, True),
        (2, 2016, True),
        (3, 2015, True),  # too few ratios: not in the evaluation universe
        (4, 2017, True),  # a test year: never read here
        (5, 2015, False),  # no 10-K text
    ]
    return pd.DataFrame(
        {
            "cik": [cik for cik, _, _ in rows],
            "fiscal_year": [year for _, year, _ in rows],
            "accession": [f"000-{cik}-{year}" for cik, year, _ in rows],
            "period_end": [pd.Timestamp(f"{year}-12-31") for _, year, _ in rows],
            "in_corpus": [in_corpus for _, _, in_corpus in rows],
            "corpus_year": [year if in_corpus else pd.NA for _, year, in_corpus in rows],
        }
    ).astype({"corpus_year": "Int64"})


def criteria_table() -> pd.DataFrame:
    filings = filings_table()
    return filings[["cik", "fiscal_year"]].assign(too_few_ratios=filings["cik"] == 3)


def load_sections(keys: set[tuple[int, int]]) -> Iterable[tuple[tuple[int, int], dict]]:
    for key in sorted(keys):
        yield key, SECTIONS[key]


class FakeClient:
    """Quotes the doubt when the chunk holds it, else answers "nothing found"."""

    model_version = "fake-model@sha256:0"

    def __init__(self, fail_after: int | None = None) -> None:
        self.prompts: list[str] = []
        self.fail_after = fail_after

    def complete(self, prompt: str, schema: dict) -> str:
        if self.fail_after is not None and len(self.prompts) >= self.fail_after:
            raise KeyboardInterrupt  # the run is killed
        self.prompts.append(prompt)
        if DOUBT in prompt:
            facts = dict.fromkeys(FIELDS, True) | {"alleviated": False}
            return json.dumps({"quote": DOUBT} | facts)
        return json.dumps({"quote": ""} | dict.fromkeys(FIELDS, False))


def run(store_dir: Path, client: FakeClient | None = None, **options) -> tuple:
    return training_run(
        filings_table(),
        criteria_table(),
        load_sections,
        client or FakeClient(),
        store_dir,
        **options,
    )


def test_reads_training_year_filings_in_the_evaluation_universe(tmp_path: Path) -> None:
    filings, _ = run(tmp_path)

    assert set(zip(filings["cik"], filings["fiscal_year"], strict=True)) == {
        (1, 2015),
        (2, 2015),
        (2, 2016),
    }


def test_each_filing_gets_its_status_and_fallback_read_mark(tmp_path: Path) -> None:
    filings, _ = run(tmp_path)

    by_filing = filings.set_index(["cik", "fiscal_year"])
    assert by_filing.loc[(1, 2015), "status"] == "flagged"
    assert by_filing.loc[(2, 2015), "status"] == "clear"
    assert by_filing.loc[(2, 2016), "status"] == "clear"
    assert not by_filing.loc[(1, 2015), "fallback_read"]
    assert by_filing.loc[(2, 2016), "fallback_read"]


def test_chunks_keep_the_quote_and_every_extracted_field(tmp_path: Path) -> None:
    _, chunks = run(tmp_path)

    alfa = chunks[(chunks["cik"] == 1)].iloc[0]
    assert alfa["quote"] == DOUBT
    assert alfa["status"] == "flagged"
    assert alfa["model_version"] == "fake-model@sha256:0"
    for field in FIELDS:
        assert field in chunks.columns
    # A fallback read is read in full: Items 8 and 15, one chunk each.
    assert sorted(chunks.loc[chunks["fiscal_year"] == 2016, "item"]) == ["15", "8"]


def test_counts_statuses_and_seconds_per_filing_by_year(tmp_path: Path) -> None:
    filings, _ = run(tmp_path)

    counts = run_counts(filings).set_index("fiscal_year")
    assert counts.loc[2015, ["filings", "flagged", "clear", "extraction_failed"]].tolist() == [
        2,
        1,
        1,
        0,
    ]
    assert counts.loc[2016, "fallback_read"] == 1
    assert counts.loc["all", "filings"] == 3
    assert counts.loc["all", "seconds_per_filing"] >= 0


def test_a_killed_run_resumes_without_rereading_finished_filings(tmp_path: Path) -> None:
    killed = FakeClient(fail_after=2)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, killed)

    resumed = FakeClient()
    filings, _ = run(tmp_path, resumed)

    assert len(filings) == 3
    # Two chunks were answered before the kill; only the rest are read again.
    assert set(resumed.prompts).isdisjoint(killed.prompts)
    assert len(killed.prompts) + len(resumed.prompts) == 4
    # Seconds are kept from the run that read the filing.
    assert filings["seconds"].notna().all()


def test_a_pilot_reads_a_fixed_sample_of_filings(tmp_path: Path) -> None:
    first, _ = run(tmp_path / "a", limit=2)
    second, _ = run(tmp_path / "b", limit=2)

    assert len(first) == 2
    assert first["accession"].tolist() == second["accession"].tolist()
