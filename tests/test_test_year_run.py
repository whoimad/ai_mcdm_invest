"""Test-year run: fiscal 2017 and 2018 only, behind a pushed pre-registration.

The model and git are fakes and the 10-K text is held in memory, so no GPU,
corpus or remote is needed. The real run is
`python -m ai_mcdm_invest.llm.test_year_run`.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest

from ai_mcdm_invest.llm.going_concern import FIELDS
from ai_mcdm_invest.llm.test_year_run import PREREGISTRATION, run_test_years

COMMIT = "a" * 40
DOUBT = (
    "The Company has suffered recurring losses from operations that raise substantial "
    "doubt about its ability to continue as a going concern."
)


def report(body: str = "The statements present fairly.") -> str:
    return (
        "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM\n"
        f"We have audited the balance sheets of Alfa.\n{body}\n"
        "/s/ Smith & Jones LLP\nMarch 14, 2018\n"
    )


# 10-K text by (cik, corpus_year): Item -> text.
SECTIONS = {
    (1, 2016): {"8": report()},  # a training year: never read here
    (1, 2017): {"8": report(DOUBT)},
    (1, 2018): {"8": report()},
}


def filings_table() -> pd.DataFrame:
    years = [2016, 2017, 2018]
    return pd.DataFrame(
        {
            "cik": [1] * len(years),
            "fiscal_year": years,
            "accession": [f"000-1-{year}" for year in years],
            "period_end": [pd.Timestamp(f"{year}-12-31") for year in years],
            "in_corpus": [True] * len(years),
            "corpus_year": years,
        }
    ).astype({"corpus_year": "Int64"})


def criteria_table() -> pd.DataFrame:
    return filings_table()[["cik", "fiscal_year"]].assign(too_few_ratios=False)


def load_sections(keys: set[tuple[int, int]]) -> Iterable[tuple[tuple[int, int], dict]]:
    for key in sorted(keys):
        yield key, SECTIONS[key]


class FakeClient:
    """Quotes the doubt when the chunk holds it, else answers "nothing found"."""

    def __init__(self, model_version: str = "fake-model@sha256:0") -> None:
        self.model_version = model_version
        self.prompts: list[str] = []

    def complete(self, prompt: str, schema: dict) -> str:
        self.prompts.append(prompt)
        if DOUBT in prompt:
            facts = dict.fromkeys(FIELDS, True) | {"alleviated": False}
            return json.dumps({"quote": DOUBT} | facts)
        return json.dumps({"quote": ""} | dict.fromkeys(FIELDS, False))


def pushed_git(repo_root: Path, args: list[str]) -> str:
    """Git where the pre-registration is committed and on a remote branch."""
    if args[0] == "status":
        return ""
    if args[0] == "log":
        return COMMIT + "\n"
    if args[0] == "branch":
        return "  origin/main\n"
    raise AssertionError(args)


def preregistered(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    (repo_root / "PREREGISTRATION.md").write_text("# Pre-registration\n")
    return repo_root


def run(tmp_path: Path, client: FakeClient | None = None, git=pushed_git) -> tuple:
    return run_test_years(
        filings_table(),
        criteria_table(),
        load_sections,
        client or FakeClient(),
        tmp_path / "store",
        repo_root=preregistered(tmp_path),
        git=git,
    )


def test_reads_only_fiscal_2017_and_2018_and_records_the_config(tmp_path: Path) -> None:
    client = FakeClient()
    filings, chunks = run(tmp_path, client)

    assert filings["fiscal_year"].tolist() == [2017, 2018]
    assert set(chunks["fiscal_year"]) == {2017, 2018}
    assert len(client.prompts) == 2
    assert filings.set_index("fiscal_year")["status"].to_dict() == {2017: "flagged", 2018: "clear"}
    assert (filings["prompt_version"] == "gc-2").all()
    assert (filings["prereg_commit"] == COMMIT).all()
    assert (chunks["prereg_commit"] == COMMIT).all()
    assert filings.loc[filings["fiscal_year"] == 2017, "keyword_flagged"].item()


def unpushed_git(repo_root: Path, args: list[str]) -> str:
    """Git where the pre-registration's commit is on no remote branch."""
    return "" if args[0] == "branch" else pushed_git(repo_root, args)


def edited_git(repo_root: Path, args: list[str]) -> str:
    """Git where the pre-registration has uncommitted changes."""
    return f" M {PREREGISTRATION}\n" if args[0] == "status" else pushed_git(repo_root, args)


@pytest.mark.parametrize("git", [unpushed_git, edited_git])
def test_refuses_to_start_without_a_committed_and_pushed_preregistration(
    tmp_path: Path, git
) -> None:
    client = FakeClient()
    with pytest.raises(RuntimeError, match=PREREGISTRATION):
        run(tmp_path, client, git=git)
    (tmp_path / "repo" / PREREGISTRATION).unlink()
    with pytest.raises(RuntimeError, match=PREREGISTRATION):
        run_test_years(
            filings_table(),
            criteria_table(),
            load_sections,
            client,
            tmp_path / "store",
            repo_root=tmp_path / "repo",
            git=pushed_git,
        )

    assert client.prompts == []
    assert not (tmp_path / "store").exists()


def test_resumes_only_under_the_same_prompt_version_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(tmp_path)
    again = FakeClient()
    filings, _ = run(tmp_path, again)
    assert again.prompts == []  # the same config: every filing is reused
    assert len(filings) == 2

    with pytest.raises(RuntimeError, match="model"):
        run(tmp_path, FakeClient("fake-model@sha256:1"))
    monkeypatch.setattr("ai_mcdm_invest.llm.test_year_run.PROMPT_VERSION", "gc-3")
    with pytest.raises(RuntimeError, match="prompt"):
        run(tmp_path)
