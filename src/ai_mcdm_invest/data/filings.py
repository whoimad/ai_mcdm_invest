"""Filings stage: the 10-K behind each joined-panel row, described.

One row per joined-panel row (filer and fiscal year). Its 10-K is chosen by
the period it reports on, never by filing order or by the corpus label:

- a row linked by its figures takes the 10-K whose SEC report date is within
  `PERIOD_TOLERANCE` of the period end of the XBRL figures it matched
  (52/53-week years end a few days off the month end);
- a row linked by ticker only has no XBRL period, so it takes the 10-K whose
  report date falls in its fiscal year (the panel names a fiscal year by the
  year it ends in), the latest if there are several.

Among 10-Ks for one period the earliest filed is the original. The corpus
record is the one whose own text names that period (see `corpus.py`). Rows
whose filer lists no such 10-K (foreign filers, funds, companies that listed
later) or whose 10-K is not in the corpus keep their row with the filing or
text columns empty, so later stages can count what they lose.
"""

import pandas as pd

from ai_mcdm_invest.data.corpus import FACT_DTYPES
from ai_mcdm_invest.data.panel import KEY_COLUMNS
from ai_mcdm_invest.data.submissions import ANNUAL_REPORT_FORMS, LATE_NOTICE_FORM

PERIOD_TOLERANCE = pd.Timedelta(days=7)
# A 10-K's late-filing notice is filed after the period ends and before the
# 10-K; without a 10-K, within a year of the period end.
NO_10K_NOTICE_WINDOW = pd.Timedelta(days=365)
# The prior year's 10-K reports on a period ending this long before.
PRIOR_PERIOD_GAP = (pd.Timedelta(days=330), pd.Timedelta(days=400))
# An Item 1A paragraph is unchanged when at least this share of its sentences
# appear, word for word, in the prior year's Item 1A.
UNCHANGED_SENTENCE_SHARE = 0.5

FACT_COLUMNS = list(FACT_DTYPES)
FILINGS_COLUMNS = [
    "cik",
    "fiscal_year",
    "ticker",
    "sector",
    "link_method",
    "period_end",
    "accession",
    "filing_date",
    "days_after_fy_end",
    "days_after_jan1",
    "december_year_end",
    "nt_10k",
    "in_corpus",
    "corpus_year",
    *FACT_COLUMNS,
    "item1a_changed_share",
    "next_year_return",
    "tail_loss",
]


def build_filings(
    joined: pd.DataFrame,
    links: pd.DataFrame,
    submissions: pd.DataFrame,
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """The filings table: one row per `joined` row (`cik`, `fiscal_year`).

    - `period_end`: the report date of the chosen 10-K (else the XBRL period)
    - `accession`, `filing_date`: the chosen 10-K, empty if none
    - `days_after_fy_end`, `days_after_jan1`: filing date minus the period end,
      and minus 1 January of the return year (`fiscal_year + 1`). A negative
      `days_after_jan1` means the 10-K was public before the return year began
    - `december_year_end`: the fiscal year ends in late December or the first
      week of January, so the return year starts just after it
    - `nt_10k`: a late-filing notice was filed for this period (empty when the
      period is unknown: a row linked by ticker whose filer lists no 10-K)
    - `in_corpus`, `corpus_year`: whether the 10-K's text is in EDGAR-CORPUS,
      and the label of its record there (to find the text again)
    - `has_item_*`, `tokens_item_*`, `tokens_total`, `gc_keyword_items`: from
      the corpus record (see `corpus.py`), empty when not in the corpus
    - `item1a_changed_share`: share of Item 1A paragraphs new or substantially
      changed since the prior year's 10-K, empty when there is no prior Item 1A
    """
    rows = joined[[*KEY_COLUMNS, "cik", "sector", "link_method", "next_year_return", "tail_loss"]]
    periods = links[[*KEY_COLUMNS, "period_end"]].rename(columns={"period_end": "xbrl_period"})
    rows = rows.merge(periods, how="left", on=KEY_COLUMNS, validate="one_to_one")
    rows["xbrl_period"] = pd.to_datetime(rows["xbrl_period"]).where(
        rows["link_method"] == "figure_match"
    )

    rows = rows.merge(_chosen_10ks(rows, submissions), how="left", on=["cik", "fiscal_year"])
    rows["period_end"] = rows["report_date"].fillna(rows["xbrl_period"])
    rows["days_after_fy_end"] = (rows["filing_date"] - rows["period_end"]).dt.days.astype("Int64")
    return_year_start = pd.to_datetime(
        (rows["fiscal_year"] + 1).astype(str) + "-01-01", format="%Y-%m-%d"
    )
    rows["days_after_jan1"] = (rows["filing_date"] - return_year_start).dt.days.astype("Int64")
    month, day = rows["period_end"].dt.month, rows["period_end"].dt.day
    rows["december_year_end"] = ((month == 12) & (day >= 20)) | ((month == 1) & (day <= 7))
    rows["december_year_end"] = (
        rows["december_year_end"].astype("boolean").where(rows["period_end"].notna())
    )
    rows["nt_10k"] = _late_notices(rows, submissions)

    rows = rows.merge(_corpus_facts(rows, corpus), how="left", on=["cik", "fiscal_year"])
    rows["in_corpus"] = rows["corpus_year"].notna()
    rows = rows.astype({"corpus_year": "Int64"} | FACT_DTYPES)
    return rows[FILINGS_COLUMNS].sort_values(["cik", "fiscal_year"], ignore_index=True)


def _chosen_10ks(rows: pd.DataFrame, submissions: pd.DataFrame) -> pd.DataFrame:
    """For each row (`cik`, `fiscal_year`), its 10-K's report date, filing date and accession."""
    reports = submissions[submissions["form"].isin(ANNUAL_REPORT_FORMS)]
    pairs = rows[["cik", "fiscal_year", "xbrl_period"]].merge(
        reports[["cik", "accession", "filing_date", "report_date"]], on="cik"
    )
    by_figures = pairs["xbrl_period"].notna()
    distance = (pairs["report_date"] - pairs["xbrl_period"]).abs()
    matches = (by_figures & (distance <= PERIOD_TOLERANCE)) | (
        ~by_figures & (pairs["report_date"].dt.year == pairs["fiscal_year"])
    )
    pairs = pairs[matches].assign(
        _distance=distance[matches].fillna(pd.Timedelta(0)),
        _report=pairs.loc[matches, "report_date"],
    )
    pairs = pairs.sort_values(
        ["cik", "fiscal_year", "_distance", "_report", "filing_date"],
        ascending=[True, True, True, False, True],
    )
    chosen = pairs.drop_duplicates(["cik", "fiscal_year"])
    return chosen[["cik", "fiscal_year", "accession", "filing_date", "report_date"]]


def _late_notices(rows: pd.DataFrame, submissions: pd.DataFrame) -> pd.Series:
    """Whether each row's filer sent a late-filing notice for its period."""
    notices = submissions.loc[submissions["form"] == LATE_NOTICE_FORM, ["cik", "filing_date"]]
    window = rows[["cik", "period_end", "filing_date"]].reset_index(names="_row")
    window["_until"] = window["filing_date"].fillna(window["period_end"] + NO_10K_NOTICE_WINDOW)
    pairs = window.merge(notices.rename(columns={"filing_date": "_notice"}), on="cik")
    inside = (pairs["_notice"] > pairs["period_end"]) & (pairs["_notice"] <= pairs["_until"])
    flagged = set(pairs.loc[inside, "_row"])
    notice = pd.Series([i in flagged for i in range(len(rows))], index=rows.index)
    return notice.astype("boolean").where(rows["period_end"].notna())  # empty: period unknown


def _corpus_facts(rows: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """For each row with a 10-K, its corpus record's label, facts and Item 1A change share."""
    records = corpus.reset_index(names="_record")
    wanted = rows.loc[rows["report_date"].notna(), ["cik", "fiscal_year", "report_date"]]
    pairs = wanted.merge(records[["cik", "_record", "corpus_year", "period_end"]], on="cik")
    distance = (pairs["period_end"] - pairs["report_date"]).abs()
    pairs = pairs[distance <= PERIOD_TOLERANCE].assign(
        _label_gap=(pairs["corpus_year"] - pairs["report_date"].dt.year).abs()
    )
    pairs = pairs.sort_values(["cik", "fiscal_year", "_label_gap", "_record"])
    chosen = pairs.drop_duplicates(["cik", "fiscal_year"])[["cik", "fiscal_year", "_record"]]

    prior = _prior_records(records)
    chosen = chosen.merge(prior, how="left", on="_record")
    paragraphs = records.set_index("_record")["item1a_paragraphs"]
    chosen["item1a_changed_share"] = [
        changed_share(paragraphs[record], paragraphs[before] if pd.notna(before) else ())
        for record, before in zip(chosen["_record"], chosen["_prior"], strict=True)
    ]
    facts = records.set_index("_record")[["corpus_year", *FACT_COLUMNS]]
    texts = chosen.join(facts, on="_record")
    return texts.drop(columns=["_record", "_prior"])


def _prior_records(records: pd.DataFrame) -> pd.DataFrame:
    """For each corpus record, the record of the same filer for the prior year, if any."""
    pairs = records[["cik", "_record", "period_end"]].merge(
        records[["cik", "_record", "period_end"]], on="cik", suffixes=("", "_before")
    )
    gap = pairs["period_end"] - pairs["period_end_before"]
    pairs = pairs[(gap >= PRIOR_PERIOD_GAP[0]) & (gap <= PRIOR_PERIOD_GAP[1])]
    pairs = pairs.assign(_off_year=(gap - pd.Timedelta(days=365)).abs())
    pairs = pairs.sort_values(["_record", "_off_year", "_record_before"])
    prior = pairs.drop_duplicates("_record")[["_record", "_record_before"]]
    return prior.rename(columns={"_record_before": "_prior"})


def changed_share(
    paragraphs: tuple[tuple[bytes, ...], ...], prior: tuple[tuple[bytes, ...], ...]
) -> float:
    """Share of `paragraphs` new or substantially changed since `prior`.

    Empty (NaN) when either year has no Item 1A paragraphs: a missing or stub
    prior year cannot tell new risks from old ones.
    """
    if not paragraphs or not prior:
        return float("nan")
    seen = {sentence for paragraph in prior for sentence in paragraph}
    changed = sum(
        sum(sentence in seen for sentence in paragraph) < UNCHANGED_SENTENCE_SHARE * len(paragraph)
        for paragraph in paragraphs
    )
    return changed / len(paragraphs)
