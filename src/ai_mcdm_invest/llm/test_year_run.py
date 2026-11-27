"""Test-year run: the one read of every fiscal 2017 and 2018 auditor's report.

The training run's code on the test years: same evaluation universe rule,
locate rule, model settings and prompt version, and the frozen keyword
benchmark on the same located blocks. It refuses to start unless
`PREREGISTRATION.md` is committed and pushed, and records that commit in every
row. Its store is apart from the training run's, which it never touches.

Run with `uv run python -m ai_mcdm_invest.llm.test_year_run`. It needs Ollama
with `qwen2.5:7b` and a GPU (about 8 hours). A killed run resumes where it
stopped, but only under the same prompt version and model digest.
"""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ai_mcdm_invest.config import REPO_ROOT, Settings
from ai_mcdm_invest.data.corpus import iter_record_sections
from ai_mcdm_invest.data.panel import TEST_YEARS
from ai_mcdm_invest.llm.going_concern import FAILED, PROMPT_VERSION, LLMClient, write_whole
from ai_mcdm_invest.llm.ollama import OllamaClient
from ai_mcdm_invest.llm.training_run import (
    KEY,
    SectionsLoader,
    going_concern_run,
    run_counts,
)
from ai_mcdm_invest.tables import write_table

PREREGISTRATION = "PREREGISTRATION.md"
NAME = "going_concern_test"

# Runs `git <args>` in a repository and returns what it prints.
Git = Callable[[Path, list[str]], str]


def run_git(repo_root: Path, args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout


def preregistration_commit(repo_root: Path, git: Git = run_git) -> str:
    """The commit that last touched `PREREGISTRATION.md`. Raises unless the file
    exists, has no uncommitted changes and that commit is on a remote branch."""
    if not (repo_root / PREREGISTRATION).is_file():
        raise RuntimeError(f"no {PREREGISTRATION} at {repo_root}: write and push it first")
    if git(repo_root, ["status", "--porcelain", "--", PREREGISTRATION]).strip():
        raise RuntimeError(f"{PREREGISTRATION} has uncommitted changes: commit and push them")
    commit = git(repo_root, ["log", "-1", "--format=%H", "--", PREREGISTRATION]).strip()
    if not commit:
        raise RuntimeError(f"{PREREGISTRATION} was never committed: commit and push it")
    if not git(repo_root, ["branch", "-r", "--contains", commit]).strip():
        raise RuntimeError(f"{PREREGISTRATION} commit {commit} is on no remote branch: push it")
    return commit


def run_test_years(
    filings: pd.DataFrame,
    criteria: pd.DataFrame,
    load_sections: SectionsLoader,
    client: LLMClient,
    store_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
    git: Git = run_git,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the going-concern facts of every test-year 10-K in the evaluation universe.

    The tables are the training run's (see `going_concern_run`), plus
    `prompt_version` and `prereg_commit` in both.
    """
    commit = preregistration_commit(repo_root, git)
    _lock_config(store_dir, client.model_version)
    filing_results, chunk_results = going_concern_run(
        filings, criteria, load_sections, client, store_dir, years=TEST_YEARS
    )
    filing_results = filing_results.assign(prompt_version=PROMPT_VERSION, prereg_commit=commit)
    chunk_results = chunk_results.assign(prereg_commit=commit)
    return filing_results, chunk_results


def _lock_config(store_dir: Path, model_version: str) -> None:
    """Record the run's prompt version and model in `store_dir`, or raise when a
    run stored there used others: one run, one config."""
    config = {"prompt_version": PROMPT_VERSION, "model_version": model_version}
    path = store_dir / "config.json"
    if path.exists():
        stored = json.loads(path.read_text())
        for field, value in config.items():
            if stored[field] != value:
                raise RuntimeError(
                    f"{store_dir} holds a run with {field} {stored[field]}, not {value}: "
                    "one run, one config"
                )
        return
    store_dir.mkdir(parents=True, exist_ok=True)
    write_whole(path, json.dumps(config))


def run_report(filing_results: pd.DataFrame, model_version: str, commit: str) -> str:
    """The run report: config, counts by year and the "extraction failed" 10-Ks."""
    counts = run_counts(filing_results)
    failed = filing_results.loc[filing_results["status"] == FAILED, [*KEY, "accession"]]
    listed = failed.to_string(index=False) if len(failed) else "none"
    return (
        f"# Going-concern run, fiscal {TEST_YEARS[0]}-{TEST_YEARS[-1]}\n\n"
        f"Model: `{model_version}`\n\nPrompt: `{PROMPT_VERSION}`\n\n"
        f"Pre-registration: `{PREREGISTRATION}` at commit `{commit}`\n\n"
        f"```\n{counts.to_string(index=False)}\n```\n\n"
        f'## "Extraction failed" 10-Ks\n\n```\n{listed}\n```\n'
    )


def main() -> None:
    settings = Settings.from_env()
    commit = preregistration_commit(REPO_ROOT)
    print(f"pre-registration: {PREREGISTRATION} at {commit}")
    client = OllamaClient()
    print(f"model: {client.model_version}, prompt: {PROMPT_VERSION}")
    filings, chunks = run_test_years(
        pd.read_parquet(settings.processed_dir / "filings.parquet"),
        pd.read_parquet(settings.processed_dir / "ratio_criteria.parquet"),
        lambda keys: iter_record_sections(settings.raw_corpus_dir, keys),
        client,
        # Apart from the training run's `going_concern_store`.
        settings.interim_dir / "going_concern_test_store",
    )
    write_table(filings, settings.processed_dir, f"{NAME}_filings")
    path = write_table(chunks, settings.processed_dir, f"{NAME}_chunks")
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report = settings.reports_dir / f"{NAME}_run.md"
    report.write_text(run_report(filings, client.model_version, commit))
    print(f"results: written to {path} and beside it; counts in {report}")
    print(run_counts(filings).to_string(index=False))


if __name__ == "__main__":
    main()
