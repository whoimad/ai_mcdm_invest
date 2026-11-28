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
from ai_mcdm_invest.mcdm.ranking import (
    GROUP_LAYER,
    PROFILES,
    criterion_weights,
    group_shares,
    weights_from_groups,
)
from ai_mcdm_invest.ml.risk_score import ML_GROUP, ML_SCORE_COLUMN, TOP_CONTRIBUTIONS
from ai_mcdm_invest.screener import (
    METHODS,
    NO_10K,
    NO_REPORT,
    NO_RUN,
    NOT_IN_CORPUS,
    NOT_READ,
    filter_rows,
    load_table,
    score,
    study_layer_row,
    tail_shares,
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
# One colour for risk and one for safety, used throughout (colour-blind safe).
RISK = "#D55E00"
SAFETY = "#0072B2"
NEUTRAL = "#8C8C8C"
STATUS_COLOURS = {FLAGGED: RISK, CLEAR: SAFETY}
# SAW is a weighted mean of percentiles, shown 0 to 100; TOPSIS a closeness, 0 to 1.
SCALE = {"SAW": 100.0, "TOPSIS": 1.0}
SCORE_FORMAT = {"SAW": "%.1f", "TOPSIS": "%.3f"}
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
        f'<span style="background:{colour};color:white;border-radius:0.4rem;'
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


def summary(shown: pd.DataFrame, reveal: bool) -> None:
    cards = st.columns(4)
    cards[0].metric("Companies shown", f"{len(shown):,}", border=True)
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


def ranking_table(shown: pd.DataFrame, method: str, reveal: bool, total: int) -> int | None:
    """The ranking; returns the selected company's cik, if a row is clicked."""
    score_label = f"Score ({method})"
    frame = pd.DataFrame(
        {
            "Rank": shown["rank"],
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
    st.caption(
        f"{len(shown)} of {total} companies. Higher is safer or better. Click a row for its card."
    )
    event = st.dataframe(
        frame,
        hide_index=True,
        height=560,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            score_label: st.column_config.ProgressColumn(
                format=SCORE_FORMAT[method], min_value=0, max_value=SCALE[method], color=SAFETY
            ),
            "ML safety (percentile)": st.column_config.NumberColumn(format="%.0f"),
            "Tail-loss probability (%)": st.column_config.NumberColumn(format="%.1f"),
            "Going concern": st.column_config.MultiselectColumn(
                options=list(STATUS_LABELS.values()),
                color=[STATUS_COLOURS.get(s, NEUTRAL) for s in STATUS_LABELS],
            ),
            "Return over next calendar year (%)": st.column_config.NumberColumn(format="%+.1f"),
        },
    )
    rows = event.selection.rows
    return int(shown["cik"].iloc[rows[0]]) if rows else None


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


def text_tab(company: pd.Series, quotes: pd.DataFrame, layers: set[str]) -> None:
    status = company["status"]
    st.markdown(
        label(STATUS_LABELS[status], STATUS_COLOURS.get(status, NEUTRAL)), unsafe_allow_html=True
    )
    if "text" not in layers:
        st.caption("The 10-K text layer is switched off: this result is not in the ranking.")
    if status == FLAGGED:
        st.markdown(
            "The auditor's report on this fiscal year expresses substantial doubt that the "
            "company can continue as a going concern. Scored 0."
        )
        mine = quotes[
            (quotes["cik"] == company["cik"]) & (quotes["fiscal_year"] == company["fiscal_year"])
        ]
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


def card(
    company: pd.Series,
    method: str,
    total: int,
    reveal: bool,
    criteria: dict[str, str],
    weights: dict[str, float],
    quotes: pd.DataFrame,
    layers: set[str],
) -> None:
    with st.container(border=True):
        st.subheader(f"{company['ticker']} · {company['filer_name']}")
        st.caption(f"{company['sector']} · fiscal {company['fiscal_year']}")
        rank_col, score_col = st.columns(2)
        rank_col.metric("Rank", f"{company['rank']} of {total}")
        score_col.metric(f"Score ({method})", score_text(company["score"], method))
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
            "set before testing; custom weights were never tested. Filters only hide rows: "
            "scores and ranks stay those of the whole year."
        )


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

# Scores and ranks are those of the whole year; filters then hide rows.
table = table.assign(score=score(table, weights, method))
rows = table[table["fiscal_year"] == year].sort_values(
    ["score", "ticker"], ascending=[False, True], na_position="last"
)
rows["rank"] = range(1, len(rows) + 1)
shown = filter_rows(rows, *filters(rows))
summary(shown, reveal)

left, right = st.columns([3, 2], gap="large")
with left:
    clicked = ranking_table(shown, method, reveal, len(rows))
# Keyed by cik, not row, so the card stays on its company as the ranking re-sorts.
if clicked is not None:
    st.session_state["card"] = clicked
with right:
    if shown.empty:
        st.info("No company matches the filters.")
    else:
        chosen = shown[shown["cik"] == st.session_state.get("card")]
        company = (chosen if len(chosen) else shown).iloc[0]
        card(company, method, len(rows), reveal, criteria, weights, quotes, layers)

about()
st.divider()
st.caption(DISCLAIMER)
