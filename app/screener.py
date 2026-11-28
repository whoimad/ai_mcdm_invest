"""The screener: the study's ranking on the two test years, with a company card.

A read-only view over the tables `make all` and `make results` read: it calls
no model and has no backend. Rankings use the package's own weights and SAW,
so each company's score is the one the study ranks it by.

Start with `make screener` or `uv run streamlit run app/screener.py`.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.criteria.ratios import REPORTED, scored_criteria
from ai_mcdm_invest.evaluation.ablation import KEY, TEXT_GROUP, TEXT_SCORE_COLUMN
from ai_mcdm_invest.evaluation.results import RUN_NAME, going_concern_criterion
from ai_mcdm_invest.llm.going_concern import CLEAR, FAILED, FLAGGED
from ai_mcdm_invest.mcdm.ranking import GROUP_LAYER, PROFILES, criterion_weights, saw
from ai_mcdm_invest.ml.risk_score import ML_GROUP, ML_SCORE_COLUMN, TOP_CONTRIBUTIONS

# The two test years; each next-year return runs over the calendar year after.
AS_OF = {2017: "Fiscal 2017, returns over 2018", 2018: "Fiscal 2018, returns over 2019"}
LAYER_LABELS = {"ratios": "Ratios", "ml": "ML risk model", "text": "10-K text (LLM)"}
GROUP_LABELS = {
    "profitability": "Profitability",
    "valuation": "Valuation",
    "balance_sheet": "Balance sheet",
    ML_GROUP: "ML risk",
    TEXT_GROUP: "Text",
}
RATIO_LABELS = {
    "roe": "Return on equity",
    "net_margin": "Net margin",
    "pe": "Price-to-earnings",
    "debt_equity": "Debt-to-equity",
    "current_ratio": "Current ratio",
    "interest_coverage": "Interest coverage",
}
# Statuses of a row the test-year run did not read, besides the run's own.
NO_REPORT = "no_report"
NOT_IN_CORPUS = "not in corpus"
NO_10K = "no 10-K found"
NOT_READ = "not read"
NO_RUN = "not available"
STATUS_LABELS = {
    FLAGGED: "Flagged",
    CLEAR: "No flags",
    FAILED: "Extraction failed",
    NO_REPORT: "No audit report",
    NOT_IN_CORPUS: "Not in corpus",
    NO_10K: "No 10-K",
    NOT_READ: "Not read",
    NO_RUN: "Not available",
}
DISCLAIMER = (
    "A ranking of scored facts under fixed weights. It does not recommend buying any "
    "stock and does not forecast returns."
)


@st.cache_data
def load(processed: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], bool]:
    """The screener table, the flagged chunks' quotes, the criteria and whether
    the test-year run's tables exist.

    One row per test-year joined-panel row with enough ratios (the rows the
    study scores), with its name, ratio scores, ML score and contributions,
    10-K status and text score.
    """
    path = Path(processed)
    directions = pd.read_parquet(path / "criterion_directions.parquet")
    criteria = {**scored_criteria(directions), ML_SCORE_COLUMN: ML_GROUP}
    ratios = pd.read_parquet(path / "ratio_criteria.parquet")
    table = ratios[ratios["fiscal_year"].isin(AS_OF) & ~ratios["too_few_ratios"]]
    panel = pd.read_parquet(path / "panel.parquet", columns=[*KEY, "next_year_return"])
    filings = pd.read_parquet(
        path / "filings.parquet", columns=[*KEY, "accession", "in_corpus", "no_report"]
    )
    ml = pd.read_parquet(
        path / "ml_criterion.parquet", columns=[*KEY, ML_SCORE_COLUMN, "tail_loss_probability"]
    )
    contributions = pd.read_parquet(path / "ml_risk.parquet").drop(columns="tail_loss_probability")
    table = (
        table.merge(panel, on=KEY, how="left", validate="one_to_one")
        .merge(filings, on=KEY, how="left", validate="one_to_one")
        .merge(ml, on=KEY, how="left", validate="one_to_one")
        .merge(contributions, on=["ticker", "fiscal_year"], how="left", validate="one_to_one")
    )
    links = pd.read_parquet(path.parent / "interim" / "company_links.parquet")
    names = links.dropna(subset=["cik"]).astype({"cik": "int64"}).drop_duplicates(KEY)
    table = table.merge(names[[*KEY, "filer_name"]], on=KEY, how="left", validate="one_to_one")
    table["filer_name"] = table["filer_name"].fillna(table["ticker"])

    run_paths = {part: path / f"{RUN_NAME}_{part}.parquet" for part in ["filings", "chunks"]}
    has_text = all(p.exists() for p in run_paths.values())
    if has_text:
        criteria[TEXT_SCORE_COLUMN] = TEXT_GROUP
        text = going_concern_criterion(pd.read_parquet(run_paths["filings"]))
        table = table.merge(
            text[[*KEY, "status", TEXT_SCORE_COLUMN]], on=KEY, how="left", validate="one_to_one"
        )
        chunks = pd.read_parquet(run_paths["chunks"])
        quotes = chunks.loc[chunks["status"] == FLAGGED, [*KEY, "chunk", "item", "quote"]]
    else:
        table = table.assign(status=pd.NA, **{TEXT_SCORE_COLUMN: float("nan")})
        quotes = pd.DataFrame(columns=[*KEY, "chunk", "item", "quote"])
    # Why a row has no result from the run: no 10-K, not in the corpus, no audit report.
    in_corpus = table["in_corpus"].fillna(False).astype(bool)
    reason = pd.Series(NOT_READ, index=table.index)
    reason = reason.mask(table["no_report"].fillna(False).astype(bool), NO_REPORT)
    reason = reason.mask(~in_corpus, NOT_IN_CORPUS).mask(table["accession"].isna(), NO_10K)
    if not has_text:
        reason = reason.mask(reason == NOT_READ, NO_RUN)
    table["status"] = table["status"].fillna(reason)
    return table.reset_index(drop=True), quotes, criteria, has_text


def pct(value: float) -> str:
    return "–" if pd.isna(value) else f"{100 * value:.0f}"


st.set_page_config(page_title="Risk-aware screener", layout="wide")
processed_dir = Settings.from_env().processed_dir
table, quotes, criteria, has_text = load(str(processed_dir))

with st.sidebar:
    st.header("Screener")
    year = st.radio("As of", list(AS_OF), format_func=AS_OF.get, index=0)
    st.caption(
        "Ranks companies on their fiscal-year data; the reveal shows the return over the "
        "next calendar year. A test on past data, not a forecast."
    )
    profile = st.radio("Weight profile", list(PROFILES), index=0)
    st.subheader("Layers")
    layers = set()
    for layer, label in LAYER_LABELS.items():
        available = layer != "text" or has_text
        help_text = None if available else "Not available: the test-year 10-K run has no tables."
        if st.toggle(label, value=available, disabled=not available, help=help_text, key=layer):
            layers.add(layer)
    rows = table[table["fiscal_year"] == year]
    sectors = sorted(rows["sector"].dropna().unique())
    sector = st.selectbox("Sector", ["All sectors", *sectors], key="sector")
    reveal = st.toggle("Reveal the return over the next calendar year", value=False)

st.title("Risk-aware screener")
st.caption(DISCLAIMER)

if not layers:
    st.info("Switch on at least one layer to rank.")
    st.stop()

# The study's weights: a layer switched off loses its weight, the rest rescaled.
weights = criterion_weights(profile, criteria, layers)
group_weight = pd.Series(weights).groupby(pd.Series(criteria)).sum()
st.markdown(
    "Weights in use: "
    + " · ".join(
        f"{GROUP_LABELS[group]} **{group_weight[group]:.0%}**"
        if group in group_weight
        else f"{GROUP_LABELS[group]} off"
        for group in PROFILES[profile]
    )
)

if sector != "All sectors":
    rows = rows[rows["sector"] == sector]
ranked = rows.assign(score=saw(rows, weights)).sort_values(
    ["score", "ticker"], ascending=[False, True]
)
ranked["rank"] = range(1, len(ranked) + 1)
ranked["ratio_score"] = saw(ranked, criterion_weights(profile, criteria, {"ratios"}))

shown = pd.DataFrame(
    {
        "Rank": ranked["rank"],
        "Ticker": ranked["ticker"],
        "Company": ranked["filer_name"],
        "Sector": ranked["sector"],
        "Score": 100 * ranked["score"],
        "Ratios": 100 * ranked["ratio_score"],
        "ML safety (percentile)": 100 * ranked[ML_SCORE_COLUMN],
        "Tail-loss probability (%)": 100 * ranked["tail_loss_probability"],
        "Going concern": ranked["status"].map(STATUS_LABELS),
    }
)
if reveal:
    shown["Return over next calendar year (%)"] = ranked["next_year_return"]

left, right = st.columns([3, 2], gap="large")
with left:
    st.caption(
        f"{len(ranked)} companies, {AS_OF[year]}. Scores are 0 to 100, higher is "
        "safer or better; ratio and ML scores are percentiles within the year."
    )
    st.dataframe(
        shown,
        hide_index=True,
        height=620,
        column_config={
            "Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
            "Ratios": st.column_config.NumberColumn(format="%.0f"),
            "ML safety (percentile)": st.column_config.NumberColumn(format="%.0f"),
            "Tail-loss probability (%)": st.column_config.NumberColumn(format="%.1f"),
            "Return over next calendar year (%)": st.column_config.NumberColumn(format="%+.1f"),
        },
    )

with right:
    # Keyed by ticker and name, not rank, so the card stays on its company as the ranking re-sorts.
    labels = ranked["ticker"] + " · " + ranked["filer_name"]
    choice = st.selectbox("Company card (in rank order)", labels.tolist(), key="card")
    if choice is None:
        st.stop()
    company = ranked[labels == choice].iloc[0]
    st.subheader(f"{company['ticker']} · {company['filer_name']}")
    st.caption(f"{company['sector']} · fiscal {year} · rank {company['rank']} of {len(ranked)}")
    st.metric("Score", pct(company["score"]))
    if reveal:
        st.caption(f"Return over the next calendar year: {company['next_year_return']:+.1f}%")

    st.markdown("**Ratios** (market-wide percentile, scored)")
    ratio_rows = []
    for column, group in criteria.items():
        if GROUP_LAYER[group] != "ratios":
            continue
        name = column.removesuffix("_score")
        status = company[f"{name}_status"]
        ratio_rows.append(
            {
                "Criterion": RATIO_LABELS.get(name, name),
                "Group": GROUP_LABELS[group],
                "Percentile": pct(company[column]),
                "Sector percentile (context, not scored)": pct(company[f"{name}_sector_score"]),
                "Read as": "" if status == REPORTED else status,
                "Weight": f"{weights[column]:.1%}" if column in weights else "off",
            }
        )
    st.dataframe(pd.DataFrame(ratio_rows), hide_index=True)

    st.markdown(
        f"**ML safety: {pct(company[ML_SCORE_COLUMN])}** (percentile, higher is safer; "
        f"tail-loss probability {100 * company['tail_loss_probability']:.1f}%)"
        + ("" if "ml" in layers else " (layer off)")
    )
    contributions = [
        (company[f"feature_{k}"], company[f"contribution_{k}"])
        for k in range(1, TOP_CONTRIBUTIONS + 1)
        if isinstance(company[f"feature_{k}"], str)
    ]
    if contributions:
        st.caption(
            f"The {len(contributions)} largest contributions to the tail-loss probability, in "
            "log-odds (SHAP values from the model's own output). A contribution that raises "
            "risk lowers the safety percentile."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": [f for f, _ in contributions],
                    "Contribution": [round(float(v), 3) for _, v in contributions],
                    "Effect": ["raises risk" if v > 0 else "lowers risk" for _, v in contributions],
                }
            ),
            hide_index=True,
        )
    else:
        st.caption("The model did not score this company.")

    st.markdown("**10-K text: going concern**" + ("" if "text" in layers else " (layer off)"))
    status = company["status"]
    if status == FLAGGED:
        st.markdown(
            "Flagged: the auditor's report on this fiscal year expresses substantial doubt "
            "that the company can continue as a going concern. Scored 0."
        )
        mine = quotes[(quotes["cik"] == company["cik"]) & (quotes["fiscal_year"] == year)]
        for item, quote in mine.sort_values("chunk")[["item", "quote"]].to_numpy():
            st.caption(f"Quoted from Item {item}:")
            st.markdown("\n".join(f"> {line}" for line in str(quote).splitlines()))
    elif status == CLEAR:
        st.markdown("No flags: the auditor's report raises no going-concern doubt. Scored 1.")
    else:
        st.markdown(
            f"Text not available ({STATUS_LABELS[status].lower()}). "
            "Nothing is scored; the company is ranked on its other criteria."
        )

st.divider()
st.caption(DISCLAIMER + " The ML layer's results are exploratory.")
