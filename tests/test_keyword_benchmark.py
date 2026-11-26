"""Keyword benchmark: the strict going-concern regex the LLM is compared against.

It reads the same located chunks as the LLM. Where a trap needs reading the
regex cannot do, the test records that it gets it wrong by design.
"""

import hashlib

import pytest

from ai_mcdm_invest.llm import keyword_benchmark
from ai_mcdm_invest.llm.keyword_benchmark import KEYWORD_VERSION, keyword_match

DOUBT = (
    "The Company has suffered recurring losses from operations that raise substantial "
    "doubt about its ability to continue as a going concern."
)
REPORT = (
    "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM\n"
    "We have audited the balance sheets of Alfa Corp. as of December 31, 2015.\n"
    "In our opinion, the statements present fairly. The accompanying financial statements "
    "have been prepared assuming that the Company will continue as a going concern. "
    "{passage} Management's plans are described in Note 1.\n"
    "/s/ Smith & Jones LLP\nMarch 14, 2016"
)


def test_a_genuine_opinion_is_matched_and_its_sentence_kept() -> None:
    assert keyword_match(REPORT.format(passage=DOUBT)) == DOUBT


# The four keyword traps (the same passages as the LLM's tests), and whether a
# strict regex can see each one. `True`: the regex flags it, which is wrong.
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
    # Wrong by design: the year of a reprinted opinion takes reading.
    "stale opinion": (STALE, True),
    "boilerplate": (BOILERPLATE, False),
    # Wrong by design: who "its" is takes reading.
    "another company's doubt": (OTHER_COMPANY, True),
    "alleviated doubt": (ALLEVIATED, False),
}


@pytest.mark.parametrize("trap", TRAPS)
def test_each_keyword_trap_and_whether_the_regex_falls_for_it(trap: str) -> None:
    passage, falls_for_it = TRAPS[trap]

    expected = passage if falls_for_it else None

    assert keyword_match(REPORT.format(passage=passage)) == expected


# Wordings of genuine opinions in the fiscal 2014–2016 blocks, shortened.
OPINIONS = [
    "The Company does not have sufficient liquidity to pay a loan maturing in 2015 raising "
    "substantial doubt about its ability to continue as a going concern.",
    "The resulting operating losses raise substantial doubts about its ability to continue as "
    "a going concern.",
    "As discussed in Note 1, there is substantial doubt about the ability of the Company to "
    "continue as a going concern.",
    "As discussed in note 1, there is substantial doubt about DelMar Pharmaceuticals, Inc.’s "
    "ability to continue as a going concern.",
    "These conditions raise substantial doubt about the Company's ability to continue to "
    "operate as a going concern.",
    "If the Company does not complete a business combination by July 28, 2017, the Company "
    "will cease all operations, thus there is a substantial doubt about its ability to "
    "continue as a going concern.",
]
# The accounting rule's own wording (ASU 2014-15), common in fallback reads.
RULE_WORDING = [
    "This update requires an entity's management to evaluate whether there are conditions or "
    "events that raise substantial doubt about the entity's ability to continue as a going "
    "concern within one year.",
    "When conditions or events raise substantial doubts about an entity’s ability to continue "
    "as a going concern, management shall disclose them.",
    "Management identified no conditions or events that would raise a substantial doubt about "
    "the Company's ability to continue as a going concern.",
    "The guidance defines management’s responsibility to assess an entity’s ability to "
    "continue as a going concern, and to provide footnote disclosures if there is substantial "
    "doubt about its ability to continue as a going concern.",
]


@pytest.mark.parametrize("opinion", OPINIONS)
def test_wordings_of_genuine_opinions_are_matched(opinion: str) -> None:
    assert keyword_match(REPORT.format(passage=opinion)) == opinion


@pytest.mark.parametrize("wording", RULE_WORDING)
def test_the_accounting_rules_own_wording_is_not_matched(wording: str) -> None:
    assert keyword_match(f"Recent accounting pronouncements. {wording} Other text.") is None


def test_a_genuine_opinion_in_the_rules_wording_is_missed_by_design() -> None:
    opinion = (
        "The Company will be unable to repay its loan, which raises substantial doubt about "
        "its ability to continue as a going concern."
    )

    assert keyword_match(REPORT.format(passage=opinion)) is None


def test_the_pattern_is_frozen_under_its_version() -> None:
    # A changed pattern must get a new KEYWORD_VERSION, and this pin with it.
    patterns = [
        keyword_benchmark._SENTENCE_BREAK.pattern,
        keyword_benchmark._DOUBT.pattern,
        keyword_benchmark._RULE_WORDING.pattern,
    ]
    digest = hashlib.sha256("\n".join(patterns).encode()).hexdigest()[:12]

    assert (KEYWORD_VERSION, digest) == ("kw-1", "59c11eec75a9")
