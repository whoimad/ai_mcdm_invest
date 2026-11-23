"""Going-concern extraction: located chunks in, verified facts and a flag out.

The LLM client is the one seam: a fake returns canned JSON, so no GPU is needed.
"""

import itertools
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.llm.going_concern import (
    FIELDS,
    extract_going_concern,
    filing_status,
    going_concern_flag,
    quote_found,
)

DOUBT = (
    "The accompanying financial statements have been prepared assuming that the Company "
    "will continue as a going concern. The Company has suffered recurring losses from "
    "operations that raise substantial doubt about its ability to continue as a going concern."
)
REPORT = (
    "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM\n"
    "We have audited the balance sheets of Alfa Corp. as of December 31, 2015.\n"
    f"{DOUBT}\n/s/ Smith & Jones LLP\nMarch 14, 2016"
)


def answer(quote: str, **fields: bool) -> str:
    """The JSON a model returns: the quote first, then the judgement fields."""
    judgement = {
        "auditor_opinion": True,
        "this_company": True,
        "current_year": True,
        "doubt_expressed": True,
        "alleviated": False,
    }
    return json.dumps({"quote": quote, **judgement | fields})


# The answer for a chunk with no going-concern passage.
NOTHING = json.dumps({"quote": ""} | dict.fromkeys(FIELDS, False))


class FakeClient:
    """Answers each prompt with `respond(prompt)`; keeps the prompts it was sent."""

    model_version = "fake-model@sha256:0"

    def __init__(self, respond: Callable[[str], str]) -> None:
        self.respond = respond
        self.prompts: list[str] = []

    def complete(self, prompt: str, schema: dict) -> str:
        self.prompts.append(prompt)
        return self.respond(prompt)


def only_row(results: pd.DataFrame) -> pd.Series:
    assert len(results) == 1
    return results.iloc[0]


def chunks(*texts: str, filing: str = "1000_2015") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filing_id": filing,
            "period_end": "2015-12-31",
            "chunk": range(len(texts)),
            "item": "8",
            "text": list(texts),
        }
    )


def test_a_verified_doubt_in_the_auditors_opinion_is_flagged(tmp_path: Path) -> None:
    client = FakeClient(lambda prompt: answer(DOUBT))

    row = only_row(extract_going_concern(chunks(REPORT), client, tmp_path))

    assert row["status"] == "flagged"
    assert row["quote"] == DOUBT
    assert row["model_version"] == "fake-model@sha256:0"


def test_only_the_auditors_current_doubt_about_this_company_is_flagged() -> None:
    combinations = itertools.product([False, True], repeat=5)
    flagged = {
        fields
        for fields in combinations
        if going_concern_flag(
            **dict(
                zip(
                    FIELDS,
                    fields,
                    strict=True,
                )
            )
        )
    }

    # Whether the doubt was alleviated is stored but never changes the score.
    assert flagged == {(True, True, True, True, False), (True, True, True, True, True)}


def test_a_quote_is_found_despite_line_breaks_and_spacing() -> None:
    chunk = "raise substantial\n  doubt about its ability to continue  as a going concern."

    assert quote_found("raise substantial doubt about its ability", chunk)
    assert quote_found("  continue as a going\nconcern. ", chunk)
    assert not quote_found("raise substantial doubt about the Company's ability", chunk)


def test_a_quote_not_in_the_chunk_is_retried_twice_then_recorded_as_failed(
    tmp_path: Path,
) -> None:
    invented = "The auditor has substantial doubt about Alfa Corp."
    client = FakeClient(lambda prompt: answer(invented))

    row = only_row(extract_going_concern(chunks(REPORT), client, tmp_path))

    assert len(client.prompts) == 3
    assert row["status"] == "extraction_failed"


def test_malformed_output_is_retried(tmp_path: Path) -> None:
    replies = iter(["The auditor doubts it.", json.dumps({"quote": DOUBT}), answer(DOUBT)])
    client = FakeClient(lambda prompt: next(replies))

    row = only_row(extract_going_concern(chunks(REPORT), client, tmp_path))

    assert len(client.prompts) == 3
    assert row["status"] == "flagged"


def test_an_empty_quote_means_no_passage_and_cannot_carry_doubt(tmp_path: Path) -> None:
    clean_report = REPORT.replace(DOUBT, "In our opinion, the statements present fairly.")
    unsupported = answer("")

    results = extract_going_concern(
        pd.concat([chunks(clean_report, filing="A"), chunks(REPORT, filing="B")]),
        FakeClient(lambda prompt: NOTHING if "fairly" in prompt else unsupported),
        tmp_path,
    )

    assert results["status"].tolist() == ["clear", "extraction_failed"]


STALE = (
    "In our report dated March 20, 2015, we expressed an opinion that the Company's recurring "
    "losses raised substantial doubt about its ability to continue as a going concern."
)
BOILERPLATE = (
    "Management is required to evaluate whether there are conditions that raise substantial "
    "doubt about the Company's ability to continue as a going concern within one year."
)
OTHER_COMPANY = (
    "Beta Inc., the Company's largest customer, disclosed substantial doubt about its "
    "ability to continue as a going concern."
)
ALLEVIATED = (
    "Management's plans to raise capital alleviate the substantial doubt about the "
    "Company's ability to continue as a going concern."
)
TRAPS = {
    "stale opinion": (STALE, {"current_year": False}),
    "boilerplate": (BOILERPLATE, {"auditor_opinion": False, "doubt_expressed": False}),
    "another company's doubt": (OTHER_COMPANY, {"auditor_opinion": False, "this_company": False}),
    "alleviated doubt": (
        ALLEVIATED,
        {"auditor_opinion": False, "doubt_expressed": False, "alleviated": True},
    ),
}


@pytest.mark.parametrize("trap", TRAPS)
def test_each_keyword_trap_scores_clear_with_its_facts_stored(trap: str, tmp_path: Path) -> None:
    passage, facts = TRAPS[trap]
    text = REPORT.replace(DOUBT, f"In our opinion, the statements present fairly. {passage}")

    row = only_row(
        extract_going_concern(
            chunks(text), FakeClient(lambda prompt: answer(passage, **facts)), tmp_path
        )
    )

    assert row["status"] == "clear"
    assert row["quote"] == passage
    assert {field: row[field] for field in facts} == facts


def test_a_cached_response_never_reaches_the_client(tmp_path: Path) -> None:
    same_text_twice = pd.concat([chunks(REPORT, filing="A"), chunks(REPORT, filing="B")])
    client = FakeClient(lambda prompt: answer(DOUBT))

    results = extract_going_concern(same_text_twice, client, tmp_path)

    assert len(client.prompts) == 1
    assert results["status"].tolist() == ["flagged", "flagged"]


def test_a_new_model_version_is_not_served_from_the_cache(tmp_path: Path) -> None:
    extract_going_concern(
        chunks(REPORT, filing="A"), FakeClient(lambda prompt: answer(DOUBT)), tmp_path
    )
    newer = FakeClient(lambda prompt: answer(DOUBT))
    newer.model_version = "fake-model@sha256:1"

    extract_going_concern(chunks(REPORT, filing="B"), newer, tmp_path)

    assert len(newer.prompts) == 1


def test_an_interrupted_run_resumes_without_repeating_finished_filings(tmp_path: Path) -> None:
    second = REPORT.replace("Alfa Corp.", "Beta Inc.")
    filings = pd.concat(
        [chunks(REPORT, filing="A"), chunks(second, "In our opinion, fine.", filing="B")]
    )

    def killed_in_filing_b(prompt: str) -> str:
        if "In our opinion, fine." in prompt:
            raise KeyboardInterrupt
        return answer(DOUBT)

    with pytest.raises(KeyboardInterrupt):
        extract_going_concern(filings, FakeClient(killed_in_filing_b), tmp_path)
    restarted = FakeClient(lambda prompt: answer(DOUBT) if "Beta" in prompt else NOTHING)
    results = extract_going_concern(filings, restarted, tmp_path)

    assert not any("Alfa Corp." in prompt for prompt in restarted.prompts)
    assert results[["filing_id", "chunk", "status"]].to_numpy().tolist() == [
        ["A", 0, "flagged"],
        ["B", 0, "flagged"],
        ["B", 1, "clear"],
    ]


def test_a_filing_is_flagged_by_any_chunk_and_failed_before_it_is_clear() -> None:
    chunk_results = pd.DataFrame(
        {
            "filing_id": ["A", "A", "B", "B", "C", "C"],
            "status": ["clear", "flagged", "clear", "extraction_failed", "clear", "clear"],
        }
    )

    statuses = filing_status(chunk_results)

    assert statuses.to_dict() == {"A": "flagged", "B": "extraction_failed", "C": "clear"}


def test_a_filing_whose_located_text_changed_is_read_again(tmp_path: Path) -> None:
    first = FakeClient(lambda prompt: NOTHING)
    extract_going_concern(chunks("In our opinion, fine.", filing="A"), first, tmp_path)
    second = FakeClient(lambda prompt: answer(DOUBT))

    results = extract_going_concern(chunks(REPORT, filing="A"), second, tmp_path)

    assert results["status"].tolist() == ["flagged"]
