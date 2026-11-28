"""The screener: the study's ranking on the two test years, with a company card.

A read-only view over the tables `make all` and `make results` read: it calls
no model and has no backend. Scores come from the package's weights, SAW and
TOPSIS (`ai_mcdm_invest.screener`); on a fixed profile each company's score is
the one the study ranks it by.

Start with `make screener` or `uv run streamlit run app/screener.py`.
"""

import html
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.criteria.ratios import REPORTED
from ai_mcdm_invest.evaluation.ablation import TEXT_GROUP
from ai_mcdm_invest.llm.going_concern import CLEAR, FAILED, FLAGGED
from ai_mcdm_invest.llm.keyword_benchmark import KEYWORD_VERSION
from ai_mcdm_invest.mcdm.ranking import (
    GROUP_LAYER,
    PROFILES,
    criterion_weights,
    group_shares,
    weights_from_groups,
)
from ai_mcdm_invest.ml.risk_score import ML_GROUP, ML_SCORE_COLUMN, TOP_CONTRIBUTIONS
from ai_mcdm_invest.screener import (
    KEYWORD_ONLY,
    METHODS,
    NO_10K,
    NO_REPORT,
    NO_RUN,
    NOT_IN_CORPUS,
    NOT_READ,
    TextEvidence,
    card_company,
    clicked_company,
    company_choices,
    edgar_url,
    filter_rows,
    load_table,
    not_in_year_line,
    parse_tickers,
    rank_label,
    rank_rows,
    score,
    shortlist_rows,
    split_known,
    study_layer_row,
    tail_shares,
    text_evidence,
)

# The two test years; each next-year return runs over the calendar year after.
AS_OF = {2017: "Fiscal 2017, returns over 2018", 2018: "Fiscal 2018, returns over 2019"}
LAYER_LABELS = {"ratios": "Ratios", "ml": "ML risk model", "text": "10-K text (LLM)"}
GROUP_LABELS = {
    "profitability": "Profitability",
    "valuation": "Valuation",
    "balance_sheet": "Balance sheet",
    ML_GROUP: "ML risk",
    TEXT_GROUP: "10-K text",
}
CUSTOM = "custom"
PROFILE_LABELS = {
    "balanced": "Balanced",
    "risk-averse": "Risk-averse",
    "value-minded": "Value-minded",
    CUSTOM: "Custom",
}
RATIO_LABELS = {
    "roe": "Return on equity",
    "net_margin": "Net margin",
    "pe": "Price-to-earnings",
    "debt_equity": "Debt-to-equity",
    "current_ratio": "Current ratio",
    "interest_coverage": "Interest coverage",
}
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
# One colour for risk and one for safety, used throughout (colour-blind safe, tuned for
# the dark theme).
RISK = "#F0834A"
SAFETY = "#56B4E9"
NEUTRAL = "#9AA3AD"
STATUS_COLOURS = {FLAGGED: RISK, CLEAR: SAFETY}
# SAW is a weighted mean of percentiles, shown 0 to 100; TOPSIS a closeness, 0 to 1.
SCALE = {"SAW": 100.0, "TOPSIS": 1.0}
SCORE_FORMAT = {"SAW": "%.1f", "TOPSIS": "%.3f"}
# The five facts the LLM gives about each quote; the flag needs the first four.
FACT_LABELS = {
    "auditor_opinion": "In the auditor's opinion",
    "this_company": "About this company",
    "current_year": "About this fiscal year",
    "doubt_expressed": "Doubt expressed",
    "alleviated": "Alleviated by management's plans",
}
MODES = {"all": "All companies", "shortlist": "My shortlist"}
DISCLAIMER = (
    "A ranking of scored facts under fixed weights. It does not recommend buying any "
    "stock and does not forecast returns. The ML layer's results are exploratory."
)


@st.cache_data
def load(processed: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], bool]:
    return load_table(Path(processed))


def pct(value: float) -> str:
    return "–" if pd.isna(value) else f"{100 * value:.0f}"


def label(text: str, colour: str) -> str:
    """A coloured label, for inline markdown."""
    return (
        f'<span style="background:{colour};color:#0F1419;border-radius:0.4rem;'
        f'padding:0.1rem 0.5rem;font-size:0.85rem">{html.escape(text)}</span>'
    )


def score_text(value: float, method: str) -> str:
    if pd.isna(value):
        return "–"
    return f"{SCALE[method] * value:{SCORE_FORMAT[method][1:]}}"


# Weights: the sliders hold 0 to 100 per group; a fixed profile fills them.
def slider_key(group: str) -> str:
    return f"weight_{group}"


def fill_sliders(profile: str) -> None:
    for group, weight in PROFILES[profile].items():
        st.session_state[slider_key(group)] = round(100 * weight)


def on_profile() -> None:
    profile = st.session_state["profile"]
    if profile != CUSTOM:
        st.session_state["base_profile"] = profile
        fill_sliders(profile)


def on_slider() -> None:
    st.session_state["profile"] = CUSTOM


def on_reset() -> None:
    base = st.session_state["base_profile"]
    st.session_state["profile"] = base
    fill_sliders(base)


def init_state() -> None:
    # Widget state is dropped when a run stops before the widget; refill it from the profile.
    base = st.session_state.setdefault("base_profile", "balanced")
    st.session_state.setdefault("profile", base)
    if any(slider_key(group) not in st.session_state for group in GROUP_LABELS):
        fill_sliders(base)
    st.session_state.setdefault("shortlist", [])
    st.session_state.setdefault("unknown", [])


# The shortlist is kept by ticker in "shortlist", so it lasts across years and
# modes; the search box's own state is dropped while it is hidden.
def on_search() -> None:
    st.session_state["shortlist"] = list(st.session_state["shortlist_box"])


def on_paste(table: pd.DataFrame) -> None:
    known, unknown = split_known(table, parse_tickers(st.session_state["paste"]))
    shortlist = list(dict.fromkeys([*st.session_state["shortlist"], *known]))
    st.session_state["shortlist"] = shortlist
    st.session_state["shortlist_box"] = shortlist
    st.session_state["unknown"] = unknown
    st.session_state["paste"] = ""


def top_bar(has_text: bool) -> tuple[int, str, set[str], bool]:
    """Year, method, layers and the reveal, in one row."""
    year_col, method_col, layer_col, reveal_col = st.columns([3, 2, 3, 2])
    year = year_col.segmented_control(
        "As of", list(AS_OF), format_func=AS_OF.get, default=2017, required=True, key="year"
    )
    method = method_col.segmented_control(
        "Method", METHODS, default="SAW", required=True, key="method"
    )
    options = [layer for layer in LAYER_LABELS if layer != "text" or has_text]
    layers = layer_col.pills(
        "Layers",
        options,
        selection_mode="multi",
        default=options,
        format_func=LAYER_LABELS.get,
        key="layers",
        help=None if has_text else "10-K text: not available, the test-year run has no tables.",
    )
    reveal_col.write("")
    reveal = reveal_col.toggle("Reveal next-year returns", value=False, key="reveal")
    return year, method, set(layers), reveal


def weights_panel(criteria: dict[str, str], layers: set[str]) -> dict[str, float] | None:
    """Profile buttons and one slider per group; the criterion weights in use."""
    profile_col, reset_col = st.columns([5, 1], vertical_alignment="bottom")
    profile_col.segmented_control(
        "Profile",
        list(PROFILE_LABELS),
        format_func=PROFILE_LABELS.get,
        required=True,
        key="profile",
        on_change=on_profile,
    )
    base = st.session_state["base_profile"]
    reset_col.button(f"Reset to {PROFILE_LABELS[base].lower()}", on_click=on_reset, key="reset")
    groups = list(GROUP_LABELS)
    for column, group in zip(st.columns(len(groups)), groups, strict=True):
        column.slider(
            GROUP_LABELS[group],
            0,
            100,
            step=5,
            key=slider_key(group),
            on_change=on_slider,
            disabled=GROUP_LAYER[group] not in layers,
        )
    if not layers:
        st.info("Switch on at least one layer to rank.")
        return None
    sliders = {group: st.session_state[slider_key(group)] for group in groups}
    try:
        shares = group_shares(sliders, criteria, layers)
    except ValueError:
        st.warning("Give at least one group in use a weight above zero.")
        return None
    st.caption(
        "In use, rescaled to 100%: "
        + " · ".join(f"{GROUP_LABELS[g]} **{shares[g]:.0%}**" for g in groups)
        + ". A layer switched off gives its groups 0; within a group, the weight is "
        "split equally over its criteria."
    )
    profile = st.session_state["profile"]
    if profile == CUSTOM:
        return weights_from_groups(sliders, criteria, layers)
    return criterion_weights(profile, criteria, layers)


def study_note(method: str, layers: set[str]) -> None:
    profile = st.session_state["profile"]
    row = study_layer_row(layers)
    if profile == CUSTOM:
        st.info("Custom weights: these are not part of the study's tests.")
    elif row is None:
        st.info(
            f"{PROFILE_LABELS[profile]} profile, but this mix of layers is not one of the "
            "study's rows (ratios only, + ML, + LLM)."
        )
    else:
        st.success(
            f"These scores match the study's: {PROFILE_LABELS[profile].lower()} profile, "
            f"{method}, layer row “{row}”."
        )


def shortlist_panel(table: pd.DataFrame) -> bool:
    """The mode switch and, in shortlist mode, the search box and the paste
    field; whether shortlist mode is on."""
    mode_col, search_col, paste_col = st.columns([2, 4, 2], vertical_alignment="bottom")
    mode = mode_col.segmented_control(
        "Show", list(MODES), format_func=MODES.get, default="all", required=True, key="mode"
    )
    if mode != "shortlist":
        return False
    choices = company_choices(table)
    if "shortlist_box" not in st.session_state:
        st.session_state["shortlist_box"] = st.session_state["shortlist"]
    search_col.multiselect(
        "Shortlist",
        list(choices),
        format_func=choices.get,
        placeholder="Search a ticker or name",
        key="shortlist_box",
        on_change=on_search,
    )
    paste_col.text_input(
        "Paste tickers",
        placeholder="AAPL, MSFT, F",
        key="paste",
        on_change=on_paste,
        args=(table,),
        help="Tickers separated by commas or spaces; press Enter to add them.",
    )
    if st.session_state["unknown"]:
        st.warning("Not found in either year: " + ", ".join(st.session_state["unknown"]) + ".")
    return True


def filters(rows: pd.DataFrame) -> tuple[list[str], list[str], str]:
    sector_col, status_col, search_col = st.columns([3, 3, 2])
    sectors = sector_col.multiselect(
        "Sector", sorted(rows["sector"].dropna().unique()), placeholder="All", key="sectors"
    )
    statuses = status_col.multiselect(
        "Going concern",
        [s for s in STATUS_LABELS if s in set(rows["status"])],
        format_func=STATUS_LABELS.get,
        placeholder="All",
        key="statuses",
    )
    search = search_col.text_input("Ticker or name", key="search")
    return sectors, statuses, search


def summary(shown: pd.DataFrame, reveal: bool, shortlist: bool) -> None:
    cards = st.columns(4)
    cards[0].metric(
        "Shortlisted companies shown" if shortlist else "Companies shown",
        f"{len(shown):,}",
        border=True,
    )
    cards[1].metric("Going concern flagged", int((shown["status"] == FLAGGED).sum()), border=True)
    if not reveal:
        return
    shares = tail_shares(shown) if len(shown) else None
    riskiest, safest = (
        ("–", "–")
        if shares is None
        else (
            f"{shares['riskiest']:.0%}",
            f"{shares['safest']:.0%}",
        )
    )
    help_text = "Share of the shown tail losers in this fifth of the shown scores; 20% by chance."
    cards[2].metric("Tail losers in riskiest fifth", riskiest, help=help_text, border=True)
    cards[3].metric("Tail losers in safest fifth", safest, help=help_text, border=True)


def ranking_table(
    shown: pd.DataFrame, method: str, reveal: bool, total: int, listed: int | None
) -> tuple[int, int] | None:
    """The ranking; returns the clicked company's (cik, fiscal year), if any.
    In shortlist mode (`listed` companies on the shortlist that year), the rank
    is the shortlist's, with the market-wide rank beside it."""
    score_label = f"Score ({method})"
    if listed is None:
        ranks = shown["rank"]
        caption = f"{len(shown)} of {total} companies."
    else:
        ranks = [
            rank_label(mine, listed, rank, total)
            for mine, rank in zip(shown["shortlist_rank"], shown["rank"], strict=True)
        ]
        caption = (
            f"{len(shown)} of {listed} shortlisted companies. Rank: in the shortlist · "
            f"among all {total:,} companies."
        )
    frame = pd.DataFrame(
        {
            "Rank": ranks,
            "Ticker": shown["ticker"],
            "Company": shown["filer_name"],
            "Sector": shown["sector"],
            score_label: SCALE[method] * shown["score"],
            "ML safety (percentile)": 100 * shown[ML_SCORE_COLUMN],
            "Tail-loss probability (%)": 100 * shown["tail_loss_probability"],
            "Going concern": shown["status"].map(lambda s: [STATUS_LABELS[s]]),
        }
    )
    if reveal:
        frame["Return over next calendar year (%)"] = shown["next_year_return"]
    st.caption(f"{caption} Higher is safer or better. Click a row for its card.")
    # A click on any cell selects its row. Without a key, the table's selection
    # starts empty whenever its rows change, so a position never points at a
    # company from an earlier filter. Widths keep every column in view.
    event = st.dataframe(
        frame,
        hide_index=True,
        height=560,
        on_select="rerun",
        selection_mode="single-cell",
        column_config={
            "Rank": st.column_config.NumberColumn(width=55, pinned=True)
            if listed is None
            else st.column_config.TextColumn(width=140, pinned=True),
            "Ticker": st.column_config.TextColumn(width=60, pinned=True),
            "Company": st.column_config.TextColumn(width=170),
            "Sector": st.column_config.TextColumn(width=110),
            score_label: st.column_config.ProgressColumn(
                format=SCORE_FORMAT[method],
                min_value=0,
                max_value=SCALE[method],
                color=SAFETY,
                width=110,
            ),
            "ML safety (percentile)": st.column_config.NumberColumn(
                "ML safety", format="%.0f", width=75, help="ML safety percentile, 0 to 100."
            ),
            "Tail-loss probability (%)": st.column_config.NumberColumn(
                "Tail-loss (%)",
                format="%.1f",
                width=85,
                help="The ML model's tail-loss probability, in %.",
            ),
            "Going concern": st.column_config.MultiselectColumn(
                options=list(STATUS_LABELS.values()),
                color=[STATUS_COLOURS.get(s, NEUTRAL) for s in STATUS_LABELS],
                width=120,
            ),
            "Return over next calendar year (%)": st.column_config.NumberColumn(
                "Next-year return (%)",
                format="%+.1f",
                width=90,
                help="Return over the next calendar year, in %.",
            ),
        },
    )
    return clicked_company(shown, event.selection.cells, event.selection.rows)


def ratios_tab(company: pd.Series, criteria: dict[str, str], weights: dict[str, float]) -> None:
    rows = []
    for column, group in criteria.items():
        if GROUP_LAYER[group] != "ratios":
            continue
        name = column.removesuffix("_score")
        weight = weights.get(column)
        status = company[f"{name}_status"]
        rows.append(
            {
                "Criterion": f"{RATIO_LABELS.get(name, name)} · "
                + ("off" if weight is None else f"{weight:.0%}"),
                "Percentile": 100 * company[column],
                "Sector": 100 * company[f"{name}_sector_score"],
                "Read as": "" if status == REPORTED else status,
            }
        )
    data = pd.DataFrame(rows)
    base = alt.Chart(data).encode(y=alt.Y("Criterion:N", sort=None, title=None))
    bars = base.mark_bar(color=SAFETY).encode(
        x=alt.X("Percentile:Q", scale=alt.Scale(domain=[0, 100]), title="Percentile"),
        tooltip=["Criterion", "Percentile", "Sector", "Read as"],
    )
    ticks = base.mark_tick(color=NEUTRAL, thickness=3, size=18).encode(x="Sector:Q")
    st.altair_chart(bars + ticks, width="stretch")
    st.caption(
        "Bars: the scored market-wide percentile; the label gives the criterion's current "
        "weight. Grey mark: the sector percentile, context, not scored."
    )
    unread = data[data["Read as"] != ""]
    for criterion, status in unread[["Criterion", "Read as"]].to_numpy():
        st.caption(f"{criterion.split(' · ')[0]}: {status}, scored at the middle (50).")


def ml_tab(company: pd.Series, layers: set[str]) -> None:
    left, right = st.columns(2)
    left.metric("ML safety (percentile)", pct(company[ML_SCORE_COLUMN]))
    probability = company["tail_loss_probability"]
    right.metric(
        "Tail-loss probability",
        "–" if pd.isna(probability) else f"{100 * probability:.1f}%",
    )
    if "ml" not in layers:
        st.caption("The ML layer is switched off: this score is not in the ranking.")
    contributions = [
        (company[f"feature_{k}"], float(company[f"contribution_{k}"]))
        for k in range(1, TOP_CONTRIBUTIONS + 1)
        if isinstance(company[f"feature_{k}"], str)
    ]
    if not contributions:
        st.caption("The model did not score this company.")
        return
    data = pd.DataFrame(contributions, columns=["Feature", "Contribution"])
    data["Effect"] = data["Contribution"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Contribution:Q", title="Contribution (log-odds)"),
            y=alt.Y("Feature:N", sort=None, title=None),
            color=alt.Color(
                "Effect:N",
                scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[RISK, SAFETY]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["Feature", alt.Tooltip("Contribution:Q", format="+.3f"), "Effect"],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"The {len(contributions)} largest contributions to the tail-loss probability (SHAP "
        "values from the model's own output). A contribution that raises risk lowers the "
        "safety percentile."
    )


def facts_line(quote: pd.Series) -> str:
    return " · ".join(
        f"{FACT_LABELS[field]}: **{'yes' if quote[field] else 'no'}**" for field in FACT_LABELS
    )


def text_tab(company: pd.Series, quotes: pd.DataFrame, layers: set[str]) -> None:
    status = company["status"]
    st.markdown(
        label(STATUS_LABELS[status], STATUS_COLOURS.get(status, NEUTRAL)), unsafe_allow_html=True
    )
    if "text" not in layers:
        st.caption("The 10-K text layer is switched off: this result is not in the ranking.")
    evidence = text_evidence(company, quotes)
    if status == FLAGGED:
        st.markdown(
            "The auditor's report on this fiscal year expresses substantial doubt that the "
            "company can continue as a going concern. Scored 0."
        )
        for _, quote in evidence.quotes.iterrows():
            st.caption(f"Quoted from Item {quote['item']}:")
            st.markdown("\n".join(f"> {line}" for line in str(quote["quote"]).splitlines()))
            st.caption(facts_line(quote))
        st.caption(
            "The LLM gives five facts about each quote; the flag needs the first four "
            "(auditor's opinion, this company, this fiscal year, doubt expressed). Each "
            "quote was found in the 10-K text by code before it was accepted."
        )
    elif status == CLEAR:
        st.markdown("No flags: the auditor's report raises no going-concern doubt. Scored 1.")
    else:
        st.markdown(
            f"Text not available ({STATUS_LABELS[status].lower()}). "
            "Nothing is scored; the company is ranked on its other criteria."
        )
    keyword_verdict(evidence)
    if isinstance(company["accession"], str):
        st.link_button(
            "Open the filing on SEC EDGAR", edgar_url(int(company["cik"]), company["accession"])
        )


def keyword_verdict(evidence: TextEvidence) -> None:
    """The keyword benchmark's verdict beside the LLM's."""
    if evidence.keyword_flag is None:
        return
    st.markdown(
        f"**Keyword benchmark ({KEYWORD_VERSION})**: "
        + ("flagged the filing." if evidence.keyword_flag else "did not flag the filing.")
    )
    if evidence.keyword_sentence:
        st.markdown(f"> {evidence.keyword_sentence}")
    if evidence.disagreement == KEYWORD_ONLY:
        st.caption("The two disagree: the keywords flag the filing, the LLM does not.")
    elif evidence.disagreement is not None:
        st.caption("The two disagree: the LLM flags the filing, the keywords do not.")


def card(
    company: pd.Series,
    method: str,
    total: int,
    reveal: bool,
    criteria: dict[str, str],
    weights: dict[str, float],
    quotes: pd.DataFrame,
    layers: set[str],
    listed: int | None,
) -> None:
    with st.container(border=True):
        st.subheader(f"{company['ticker']} · {company['filer_name']}")
        st.caption(f"{company['sector']} · fiscal {company['fiscal_year']}")
        rank_col, score_col = st.columns(2)
        if listed is None:
            rank_col.metric("Rank", f"{company['rank']} of {total}")
        else:
            rank_col.metric("Rank in shortlist", f"{company['shortlist_rank']} of {listed}")
        score_col.metric(f"Score ({method})", score_text(company["score"], method))
        if listed is not None:
            st.caption(f"Rank among all companies: {company['rank']:,} of {total:,}.")
        if reveal:
            st.caption(f"Return over the next calendar year: {company['next_year_return']:+.1f}%")
        ratios, ml, text = st.tabs(["Ratios", "ML risk", "10-K text"])
        with ratios:
            ratios_tab(company, criteria, weights)
        with ml:
            ml_tab(company, layers)
        with text:
            text_tab(company, quotes, layers)


def about() -> None:
    with st.expander("About the method"):
        st.markdown(
            "Each criterion is a percentile within the fiscal year, higher is safer or "
            "better: six ratios, the ML safety percentile and the auditor's going-concern "
            "flag read by the LLM (clear 1, flagged 0).\n\n"
            "- **SAW** (the study's main ranker): the weighted sum of the percentiles, "
            "shown from 0 to 100.\n"
            "- **TOPSIS** (the study's robustness check): the closeness to the ideal company, "
            "between 0 and 1; the ideal and the scale are taken from the study's universe.\n\n"
            "Group weights are split equally over their criteria; a layer switched off "
            "gives its groups no weight and the rest are rescaled. The fixed profiles were "
            "set before testing; custom weights were never tested. A shortlist only picks "
            "rows and filters only hide rows: scores and market-wide ranks stay those of "
            "the whole year, and the shortlist's own rank follows the same order."
        )


def company_key(company: pd.Series) -> tuple[int, int]:
    return int(company["cik"]), int(company["fiscal_year"])


st.set_page_config(page_title="Risk-aware screener", layout="wide")
table, quotes, criteria, has_text = load(str(Settings.from_env().processed_dir))
init_state()

st.title("Risk-aware screener")
with st.container(border=True):
    year, method, layers, reveal = top_bar(has_text)
    weights = weights_panel(criteria, layers)
if weights is None:
    st.stop()
study_note(method, layers)

# Scores and ranks are those of the whole year; the shortlist only picks rows,
# and filters then hide rows.
table = table.assign(score=score(table, weights, method))
rows = rank_rows(table[table["fiscal_year"] == year])
on_shortlist = shortlist_panel(table)
listed, missing = (
    shortlist_rows(rows, st.session_state["shortlist"]) if on_shortlist else (rows, [])
)
shown = filter_rows(listed, *filters(rows))
summary(shown, reveal, on_shortlist)
size = len(listed) if on_shortlist else None

left, right = st.columns([3, 2], gap="large")
with left:
    if on_shortlist and not st.session_state["shortlist"]:
        st.info("Add companies to the shortlist: search a ticker or name, or paste tickers.")
    clicked = ranking_table(shown, method, reveal, len(rows), size)
    if missing:
        st.caption(not_in_year_line(missing, year))
# Keyed by (cik, fiscal year), not row, so the card stays on its company as the
# ranking re-sorts; it is always one of the rows shown.
if clicked is not None:
    st.session_state["card"] = clicked
with right:
    chosen = st.session_state.get("card")
    company = card_company(shown, chosen)
    if company is None:
        st.info(
            "No shortlisted company to show." if on_shortlist else "No company matches the filters."
        )
    else:
        if chosen is not None and company_key(company) != chosen:
            st.caption("The company you chose is not among the rows shown: this is the first row.")
        card(company, method, len(rows), reveal, criteria, weights, quotes, layers, size)

about()
st.divider()
st.caption(DISCLAIMER)
