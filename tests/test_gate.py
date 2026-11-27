"""Go/no-go gate: canaries planted in training-year 10-Ks, checks out.

The model is a fake and the 10-K text is held in memory, so no GPU or corpus
is needed. The real gate is `python -m ai_mcdm_invest.llm.gate`.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ai_mcdm_invest.llm.gate import (
    CANARY_WORDINGS,
    canary_paragraph,
    gate_report,
    plant_canary,
    run_gate,
)
from ai_mcdm_invest.llm.going_concern import FIELDS
from ai_mcdm_invest.llm.locate import locate_auditor_reports

PARAGRAPH = "These conditions raise substantial doubt about its ability to continue."


def report(company: str, body: str = "") -> str:
    return (
        "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM\n"
        f"We have audited the balance sheets of {company}. In our opinion, the "
        "financial statements referred to above present fairly, in all material respects, "
        f"the financial position of {company}. {body}\n"
        "/s/ Smith & Jones LLP\nMarch 14, 2016\n"
    )


def test_a_canary_lands_inside_the_located_report_after_the_opinion() -> None:
    sections = {"8": "Item 8. Statements.\n" + report("Alfa") + "Notes to the statements."}

    planted = plant_canary(sections, PARAGRAPH)

    assert planted is not None
    (chunk,) = locate_auditor_reports(planted).chunks
    assert PARAGRAPH in chunk.text
    assert chunk.text.index("present fairly") < chunk.text.index(PARAGRAPH)
    assert chunk.text.index(PARAGRAPH) < chunk.text.index("/s/ Smith & Jones LLP")
    assert planted["8"].replace(PARAGRAPH, "").split() == sections["8"].split()


def test_no_canary_without_a_located_report_or_an_opinion_sentence() -> None:
    fallback = {"8": "We have audited the balance sheets of Alfa.", "15": "Exhibits."}
    no_opinion = {
        "8": "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM\n"
        "We have audited the balance sheets of Alfa.\n/s/ Smith & Jones LLP\nMarch 14, 2016\n"
    }

    assert plant_canary(fallback, PARAGRAPH) is None
    assert plant_canary(no_opinion, PARAGRAPH) is None


def test_a_canary_that_would_change_where_the_report_ends_is_not_planted() -> None:
    sections = {"8": report("Alfa")}

    # A line that starts with a date ends the report early: the rest is cut off.
    assert plant_canary(sections, "March 1, 2016\nThe rest.") is None


def test_every_canary_wording_can_be_planted() -> None:
    sections = {"8": report("Alfa")}

    for kind, wordings in CANARY_WORDINGS.items():
        for variant in range(len(wordings)):
            paragraph = canary_paragraph(kind, variant, "2015-12-31")
            assert plant_canary(sections, paragraph) is not None, (kind, variant)


def test_a_stale_canary_is_about_an_earlier_fiscal_year() -> None:
    for variant in range(len(CANARY_WORDINGS["stale"])):
        paragraph = canary_paragraph("stale", variant, "2015-12-31")

        assert "December 31, 2014" in paragraph
        assert "December 31, 2015" not in paragraph


# The training-year 10-Ks the gate draws on.
DOUBT = (
    "The Company has suffered recurring losses from operations that raise substantial "
    "doubt about its ability to continue as a going concern."
)
HOSTS = range(1, 9)  # clear, located: canary hosts; cik 1 is the largest
SECTIONS = {
    **{(cik, 2015): {"8": report(f"Company {cik}")} for cik in HOSTS},
    (11, 2015): {"8": report("Kilo", DOUBT)},
    (12, 2015): {"8": report("Lima", DOUBT)},
    # Clear in its located report, but doubt in the notes after it: a locate
    # miss. Its report lost its opinion sentence, so it takes no canary.
    (13, 2015): {
        "8": "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM\n"
        "We have audited the balance sheets of Mike.\n/s/ Smith & Jones LLP\nMarch 14, 2016\n"
        "Note 2. Going concern.\n" + DOUBT
    },
}
STATUS = {cik: "clear" for cik in HOSTS} | {11: "flagged", 12: "flagged", 13: "clear"}


def training_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cik": list(STATUS),
            "fiscal_year": 2015,
            "accession": [f"000-{cik}-2015" for cik in STATUS],
            "status": list(STATUS.values()),
            "fallback_read": False,
            "keyword_flagged": [status == "flagged" for status in STATUS.values()],
        }
    )


def filings_table() -> pd.DataFrame:
    return training_table()[["accession"]].assign(
        period_end=pd.Timestamp("2015-12-31"), corpus_year=2015
    )


def market_caps() -> pd.DataFrame:
    return training_table()[["cik", "fiscal_year"]].assign(
        market_cap=lambda table: 1e9 / table["cik"]
    )


def load_sections(keys: set[tuple[int, int]]) -> Iterable[tuple[tuple[int, int], dict]]:
    for key in sorted(keys):
        yield key, SECTIONS[key]


def facts(quote: str, **fields: bool) -> str:
    true = dict.fromkeys(FIELDS, True) | {"alleviated": False}
    return json.dumps({"quote": quote} | true | fields)


NOTHING = json.dumps({"quote": ""} | dict.fromkeys(FIELDS, False))
# What a reader who gets every canary right answers about each kind.
RIGHT = {
    "genuine": {},
    "boilerplate": {"auditor_opinion": False, "doubt_expressed": False},
    "stale": {"current_year": False},
    "other_company": {"this_company": False},
}


class OracleClient:
    """Answers each canary as it should be answered, and quotes `DOUBT` when the
    text holds it. `wrong` lists the kinds it gets wrong: it calls them genuine."""

    model_version = "fake-model@sha256:0"

    def __init__(self, wrong: Iterable[str] = ()) -> None:
        self.wrong = set(wrong)
        self.prompts: list[str] = []

    def complete(self, prompt: str, schema: dict) -> str:
        self.prompts.append(prompt)
        for kind, wordings in CANARY_WORDINGS.items():
            for variant in range(len(wordings)):
                paragraph = canary_paragraph(kind, variant, "2015-12-31")
                if paragraph in prompt:
                    return facts(paragraph, **({} if kind in self.wrong else RIGHT[kind]))
        return facts(DOUBT) if DOUBT in prompt else NOTHING


def gate(store_dir: Path, client: OracleClient | None = None):
    return run_gate(
        training_table(),
        filings_table(),
        market_caps(),
        load_sections,
        client or OracleClient(),
        store_dir,
        canaries=8,
        well_known=4,
        stable=4,
        full_reads=20,  # more than there are: every located 10-K
    )


def test_canaries_of_each_kind_are_planted_in_well_known_and_other_hosts(
    tmp_path: Path,
) -> None:
    canaries = gate(tmp_path).canaries

    assert len(canaries) == 8
    assert set(canaries["cik"]) == set(HOSTS)
    assert set(canaries.loc[canaries["well_known"], "cik"]) == {1, 2, 3, 4}
    for kind in CANARY_WORDINGS:
        of_kind = canaries[canaries["kind"] == kind]
        assert sorted(of_kind["well_known"]) == [False, True]
    assert canaries.set_index("kind")["expected"].to_dict() == {
        "genuine": "flagged",
        "boilerplate": "clear",
        "stale": "clear",
        "other_company": "clear",
    }


def test_a_reader_that_gets_every_canary_right_passes_the_canary_check(
    tmp_path: Path,
) -> None:
    result = gate(tmp_path)

    assert result.canaries["correct"].all()
    check = result.checks.set_index("check").loc["canaries"]
    assert check["value"] == "8 of 8 correct (100.0%)"
    assert check["passed"]


def test_a_reader_that_falls_for_stale_opinions_fails_the_canary_check(
    tmp_path: Path,
) -> None:
    result = gate(tmp_path, OracleClient(wrong=["stale"]))

    wrong = result.canaries[~result.canaries["correct"]]
    assert set(wrong["kind"]) == {"stale"}
    assert (wrong["status"] == "flagged").all()
    check = result.checks.set_index("check").loc["canaries"]
    assert check["value"] == "6 of 8 correct (75.0%)"
    assert not check["passed"]


class FlipClient(OracleClient):
    """Quotes `DOUBT` only every other time it is asked: an unstable model."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: dict[str, int] = {}

    def complete(self, prompt: str, schema: dict) -> str:
        reply = super().complete(prompt, schema)
        if DOUBT not in prompt:
            return reply
        self.asked[prompt] = self.asked.get(prompt, 0) + 1
        return reply if self.asked[prompt] % 2 else NOTHING


def test_a_stable_reader_gives_identical_labels_in_three_runs(tmp_path: Path) -> None:
    result = gate(tmp_path)

    stability = result.stability
    assert len(stability) == 4
    assert set(stability.loc[stability["training_status"] == "flagged", "cik"]) == {11, 12}
    assert stability["identical"].all()
    check = result.checks.set_index("check").loc["stability"]
    assert check["value"] == "4 of 4 10-Ks identical in 3 runs"
    assert check["passed"]


def test_an_unstable_reader_fails_even_though_its_answers_are_cached(tmp_path: Path) -> None:
    result = gate(tmp_path, FlipClient())

    stability = result.stability.set_index("cik")
    assert stability.loc[11, ["run_1", "run_2", "run_3"]].tolist() == [
        "flagged",
        "clear",
        "flagged",
    ]
    assert not stability.loc[11, "identical"]
    assert not result.checks.set_index("check").loc["stability", "passed"]


def test_the_full_read_measures_how_often_locate_misses_a_doubt(tmp_path: Path) -> None:
    result = gate(tmp_path)

    full_reads = result.full_reads.set_index("cik")
    assert set(full_reads.index) == {*HOSTS, 11, 12, 13}
    assert full_reads.loc[13, ["located_status", "full_status"]].tolist() == ["clear", "flagged"]
    assert full_reads.loc[13, "full_quote"] == DOUBT
    assert full_reads.index[full_reads["missed"]].tolist() == [13]
    check = result.checks.set_index("check").loc["locate_miss_rate"]
    assert check["value"] == "1 of 11 10-Ks (9.1%) flagged only when read in full"
    assert check["passed"] is None  # measured and reported, no pass bar


def test_every_quote_the_gate_reads_is_checked_against_its_chunk(tmp_path: Path) -> None:
    result = gate(tmp_path)

    check = result.checks.set_index("check").loc["quotes"]
    # 8 canaries, 2 flagged 10-Ks in each of 3 stability runs, 3 flagged full reads.
    assert check["value"] == "17 of 17 quotes found in their chunk"
    assert check["passed"]


def test_the_report_states_pass_or_fail_per_check_and_overall(tmp_path: Path) -> None:
    passed = gate_report(gate(tmp_path / "a"), "fake-model@sha256:0")
    failed = gate_report(gate(tmp_path / "b", OracleClient(wrong=["stale"])), "fake")

    assert "fake-model@sha256:0" in passed
    assert "**Gate: PASS**" in passed
    assert "| canaries | 8 of 8 correct (100.0%) | at least 95% correct | PASS |" in passed
    assert "| locate_miss_rate | 1 of 11 10-Ks (9.1%) flagged only when read in full" in passed
    assert "| measured and reported | reported |" in passed
    assert "**Gate: FAIL**" in failed
    assert "| canaries | 6 of 8 correct (75.0%) | at least 95% correct | FAIL |" in failed


def test_the_report_breaks_canaries_down_by_kind_and_by_well_known_host(
    tmp_path: Path,
) -> None:
    text = gate_report(gate(tmp_path, OracleClient(wrong=["stale"])), "fake")

    assert "| stale | 2 | 0 | 0.0% |" in text
    assert "| genuine | 2 | 2 | 100.0% |" in text
    assert "| well-known companies | 4 | 3 | 75.0% |" in text
    assert "| other companies | 4 | 3 | 75.0% |" in text
