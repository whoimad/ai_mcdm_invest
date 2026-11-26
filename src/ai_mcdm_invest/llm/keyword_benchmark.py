"""Keyword benchmark: a strict going-concern regex, the non-LLM rival (decision 6.6).

It reads the same located chunks as the LLM (see `locate.py`), one sentence at
a time. A sentence matches when it says there is substantial doubt about a
named company's ability to continue as a going concern ("its", "the
Company's", "the ability of the Company", "DelMar Pharmaceuticals, Inc.'s"),
and holds none of the words of the accounting rule's own wording ("whether",
"would", "evaluate", "assess", "alleviate", "no conditions", "will be"). Doubt
about "an entity" or "the entity" is the rule's wording too and never matches.
So a genuine opinion that uses one of those words is missed: strict by design.

A chunk is flagged when any sentence matches; a 10-K when any chunk is. The
regex cannot tell an opinion reprinted from an earlier year, nor another
company's doubt ("its" may be Beta Inc.), so it flags both: wrong by design.

Tuned on the fiscal 2014–2016 chunks only, against the LLM's training-run
flags (there are no hand labels), then frozen as `KEYWORD_VERSION`. Any change
to the pattern is a new version; a test pins the pattern to the version. The
README gives its training-year counts.
"""

import re

KEYWORD_VERSION = "kw-1"

# A sentence ends at a period, semicolon or colon followed by a capital letter
# or by the auditor's signature ("/s/"), so "Inc." and "U.S." do not end it; or
# at a bullet.
_SENTENCE_BREAK = re.compile(r"(?<=[.;:])\s+(?=[\"“(]?[A-Z]|/s/)|\s*[•●]\s*")
_DOUBT = re.compile(
    r"substantial\s+doubts?\s+(?:exists?\s+)?(?:about|as\s+to|regarding|concerning|on)\s+"
    r"(?!(?:an?|the)\s+(?:entity|organization|customer)\b)[^;]{0,120}?ability\s+"
    r"(?:of\s+[^;]{1,80}?\s+)?to\s+continue\s+(?:to\s+operate\s+)?as\s+a\s+going[\s-]+concern",
    re.IGNORECASE,
)
_RULE_WORDING = re.compile(
    r"\b(?:whether|would|evaluat\w*|assess\w*|alleviat\w*|no\s+conditions|will\s+be)\b",
    re.IGNORECASE,
)


def keyword_match(text: str) -> str | None:
    """The first sentence of `text` the regex flags, or None when none does."""
    for sentence in _SENTENCE_BREAK.split(text):
        if _DOUBT.search(sentence) and not _RULE_WORDING.search(sentence):
            return sentence.strip()
    return None
