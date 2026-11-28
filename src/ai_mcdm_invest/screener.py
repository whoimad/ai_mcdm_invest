"""The screener's data and logic, kept apart from its page (`app/screener.py`).

The table is one row per test-year company with enough ratios; `in_study`
marks the study's evaluation universe. Scores come from the package's SAW and
TOPSIS; filters only hide rows.
"""

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from ai_mcdm_invest.criteria.ratios import scored_criteria
from ai_mcdm_invest.evaluation.ablation import (
    KEY,
    LAYERS,
    TEXT_GROUP,
    TEXT_SCORE_COLUMN,
    ranking_metrics,
)
from ai_mcdm_invest.evaluation.results import RUN_NAME, going_concern_criterion
from ai_mcdm_invest.llm.going_concern import CLEAR, FIELDS, FLAGGED
from ai_mcdm_invest.mcdm.ranking import saw, topsis
from ai_mcdm_invest.ml.risk_score import ML_GROUP, ML_SCORE_COLUMN

TEST_YEARS = [2017, 2018]
METHODS = ["SAW", "TOPSIS"]
# Statuses of a row the test-year run did not read, besides the run's own.
NO_REPORT = "no_report"
NOT_IN_CORPUS = "not in corpus"
NO_10K = "no 10-K found"
NOT_READ = "not read"
NO_RUN = "not available"
# The two ways the LLM and the keyword benchmark can disagree on a filing.
KEYWORD_ONLY = "keyword only"
LLM_ONLY = "LLM only"
QUOTE_COLUMNS = [*KEY, "chunk", "item", "status", "quote", *FIELDS, "keyword_match"]


def load_table(processed: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], bool]:
    """The screener table, the chunks behind each flag, the criteria (column
    to group) and whether the test-year run's tables exist.

    One row per test-year joined-panel row with enough ratios, with its name,
    ratio scores, ML score and contributions, 10-K status, text score, kw-1's
    flag, next-year return, tail loss and `in_study`. The chunks are those the
    LLM flagged or kw-1 matched: their quote, the LLM's facts and kw-1's sentence.
    """
    directions = pd.read_parquet(processed / "criterion_directions.parquet")
    criteria = {**scored_criteria(directions), ML_SCORE_COLUMN: ML_GROUP}
    ratios = pd.read_parquet(processed / "ratio_criteria.parquet")
    table = ratios[ratios["fiscal_year"].isin(TEST_YEARS) & ~ratios["too_few_ratios"]]
    panel = pd.read_parquet(
        processed / "panel.parquet", columns=[*KEY, "next_year_return", "tail_loss"]
    )
    filings = pd.read_parquet(
        processed / "filings.parquet", columns=[*KEY, "accession", "in_corpus", "no_report"]
    )
    ml = pd.read_parquet(
        processed / "ml_criterion.parquet",
        columns=[*KEY, ML_SCORE_COLUMN, "tail_loss_probability"],
    )
    contributions = pd.read_parquet(processed / "ml_risk.parquet").drop(
        columns="tail_loss_probability"
    )
    table = (
        table.merge(panel, on=KEY, how="left", validate="one_to_one")
        .merge(filings, on=KEY, how="left", validate="one_to_one")
        .merge(ml, on=KEY, how="left", validate="one_to_one")
        .merge(contributions, on=["ticker", "fiscal_year"], how="left", validate="one_to_one")
    )
    links = pd.read_parquet(processed.parent / "interim" / "company_links.parquet")
    names = links.dropna(subset=["cik"]).astype({"cik": "int64"}).drop_duplicates(KEY)
    table = table.merge(names[[*KEY, "filer_name"]], on=KEY, how="left", validate="one_to_one")
    table["filer_name"] = table["filer_name"].fillna(table["ticker"])

    run_paths = {part: processed / f"{RUN_NAME}_{part}.parquet" for part in ["filings", "chunks"]}
    has_text = all(p.exists() for p in run_paths.values())
    if has_text:
        criteria[TEXT_SCORE_COLUMN] = TEXT_GROUP
        text = going_concern_criterion(pd.read_parquet(run_paths["filings"]))
        table = table.merge(
            text[[*KEY, "status", "keyword_flag", TEXT_SCORE_COLUMN]],
            on=KEY,
            how="left",
            validate="one_to_one",
        )
        chunks = pd.read_parquet(run_paths["chunks"])
        kept = (chunks["status"] == FLAGGED) | chunks["keyword_match"].notna()
        quotes = chunks.loc[kept, QUOTE_COLUMNS].reset_index(drop=True)
    else:
        table = table.assign(status=pd.NA, keyword_flag=pd.NA, **{TEXT_SCORE_COLUMN: float("nan")})
        quotes = pd.DataFrame(columns=QUOTE_COLUMNS)
    # The study's evaluation universe (decision 5.2): readable 10-K text with an audit report.
    in_corpus = table["in_corpus"].fillna(False).astype(bool)
    no_report = table["no_report"].fillna(True).astype(bool)
    table["in_study"] = in_corpus & ~no_report
    # Why a row has no result from the run: no 10-K, not in the corpus, no audit report.
    reason = pd.Series(NOT_READ, index=table.index)
    reason = reason.mask(table["no_report"].fillna(False).astype(bool), NO_REPORT)
    reason = reason.mask(~in_corpus, NOT_IN_CORPUS).mask(table["accession"].isna(), NO_10K)
    if not has_text:
        reason = reason.mask(reason == NOT_READ, NO_RUN)
    table["status"] = table["status"].fillna(reason)
    return table.reset_index(drop=True), quotes, criteria, has_text


def score(table: pd.DataFrame, weights: Mapping[str, float], method: str) -> pd.Series:
    """Each company's score under `weights`, by SAW or TOPSIS.

    TOPSIS takes its scale, ideal and worst from the study's universe, both
    test years together, as the study does; so a universe company scores
    exactly as in the study, and the others are measured on the same scale.
    """
    if method == "SAW":
        return saw(table, weights)
    return topsis(table, weights, reference=table[table["in_study"]])


def filter_rows(
    rows: pd.DataFrame, sectors: Collection[str], statuses: Collection[str], search: str
) -> pd.DataFrame:
    """The rows in `sectors` with a 10-K status in `statuses` whose ticker or
    name contains `search` (any case). An empty filter keeps every row."""
    kept = pd.Series(True, index=rows.index)
    if sectors:
        kept &= rows["sector"].isin(sectors)
    if statuses:
        kept &= rows["status"].isin(statuses)
    text = search.strip().lower()
    if text:
        kept &= rows["ticker"].str.lower().str.contains(text, regex=False) | rows[
            "filer_name"
        ].str.lower().str.contains(text, regex=False)
    return rows[kept]


def tail_shares(rows: pd.DataFrame) -> dict[str, float] | None:
    """The share of the tail losers in the riskiest and in the safest fifth of
    `rows`' scores, cut as the study cuts them; None without a tail loser."""
    if not rows["tail_loss"].astype(bool).any():
        return None
    metrics = ranking_metrics(rows)
    return {
        "riskiest": metrics["riskiest_fifth_tail_share"],
        "safest": metrics["safest_fifth_tail_share"],
    }


def study_layer_row(layers: Collection[str]) -> str | None:
    """The study's ablation row with exactly these layers on, if there is one."""
    return next((row for row, on in LAYERS.items() if on == set(layers)), None)


def edgar_url(cik: int, accession: str) -> str:
    """The filing's folder on SEC EDGAR; it opens only in the reader's browser."""
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"


def card_company(shown: pd.DataFrame, chosen: tuple[int, int] | None) -> pd.Series | None:
    """The company the card shows: the chosen one (cik, fiscal year) if it is
    among the rows shown, else the first row shown, else none."""
    if shown.empty:
        return None
    if chosen is not None:
        match = shown[(shown["cik"] == chosen[0]) & (shown["fiscal_year"] == chosen[1])]
        if len(match):
            return match.iloc[0]
    return shown.iloc[0]


def clicked_company(
    shown: pd.DataFrame, cells: Sequence[tuple[int, str]], rows: Sequence[int]
) -> tuple[int, int] | None:
    """The company (cik, fiscal year) of the clicked cell's row, else of the
    ticked row; positions count the rows shown."""
    positions = [cell[0] for cell in cells] or list(rows)
    if not positions:
        return None
    company = shown.iloc[positions[0]]
    return int(company["cik"]), int(company["fiscal_year"])


class TextEvidence(NamedTuple):
    """What the card's 10-K text tab shows for one filing."""

    quotes: pd.DataFrame  # the LLM's flagged chunks, in order, with their facts
    keyword_flag: bool | None  # whether kw-1 flagged the filing; None if not read
    keyword_sentence: str | None  # kw-1's first matching sentence
    disagreement: str | None  # KEYWORD_ONLY, LLM_ONLY or None


def text_evidence(company: pd.Series, quotes: pd.DataFrame) -> TextEvidence:
    """The LLM's quotes and the keyword benchmark's verdict on the company's
    filing, and how they disagree when the LLM gave a verdict."""
    mine = quotes[
        (quotes["cik"] == company["cik"]) & (quotes["fiscal_year"] == company["fiscal_year"])
    ].sort_values("chunk")
    flag = company["keyword_flag"]
    keyword_flag = None if pd.isna(flag) else bool(flag)
    sentences = mine["keyword_match"].dropna()
    llm_flag = {FLAGGED: True, CLEAR: False}.get(company["status"])
    disagreement = None
    if llm_flag is not None and keyword_flag is not None and llm_flag != keyword_flag:
        disagreement = KEYWORD_ONLY if keyword_flag else LLM_ONLY
    return TextEvidence(
        quotes=mine[mine["status"] == FLAGGED],
        keyword_flag=keyword_flag,
        keyword_sentence=sentences.iloc[0] if len(sentences) else None,
        disagreement=disagreement,
    )
