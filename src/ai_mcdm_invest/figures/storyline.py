"""The four storyline figures: the report's argument before the system appears.

1. `join_funnel`: the join keeps the tail losers a ticker join would lose.
2. `leverage_tail_loss`: across the whole market, leverage points the wrong
   way, because the most indebted sectors are the safest.
3. `section_lengths`: many Items 1A, 7 and 8 do not fit a 32k-token window.
4. `going_concern_items`: "going concern" appears in many Items besides the
   auditor's report, so a keyword search is not enough.

Each figure is drawn from the pipeline's tables only, and written as PDF (for
the report) and PNG (to look at), with its caption in `captions.md`. Run with
`make figures` to redraw from the tables on disk, or as the last step of
`make all`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.join_report import join_steps
from ai_mcdm_invest.data.panel import TAIL_SHARE

# Qwen2.5's 32k context, in tokens; the filings table estimates tokens as characters ÷ 4.
CONTEXT_WINDOW_TOKENS = 32_768
# Risk factors, management's discussion, financial statements: the Items the LLM reads.
WINDOW_ITEMS = ["1A", "7", "8"]
# Histogram range of section lengths, as powers of ten tokens; longer or shorter
# sections are drawn in the end bins.
LENGTH_RANGE = (2, 6)
# The leverage measure among the six textbook ratios (decision 4.1), cut in fifths.
LEVERAGE = "debtEquityRatio"
QUINTILES = 5
QUINTILE_NAMES = ["least indebted", "second", "third", "fourth", "most indebted"]

# Sector labels that would overlap a neighbour in the leverage scatter.
LABEL_OFFSETS = {
    "Industrials": ((5, -9), "left"),
    "Consumer Defensive": ((-5, -9), "right"),
    "Utilities": ((-5, -10), "right"),
    "Real Estate": ((0, 6), "center"),
}
# Items with fewer going-concern mentions than this share are drawn together.
MINOR_ITEM_SHARE = 0.01

# Items in the order a 10-K lists them, for the going-concern figure.
ITEM_ORDER = [
    *["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B"],
    *["10", "11", "12", "13", "14", "15"],
]
# Where the auditor's report and the financial statements sit.
STATEMENT_ITEMS = {"8", "15"}

# Source citations for the captions.
KAGGLE = 'Kaggle "200+ Financial Indicators of US Stocks"'
SEC = "SEC submissions and XBRL frames"
CORPUS = "EDGAR-CORPUS 10-K sections"

# Reference palette (dataviz skill), light mode for print.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e3df"


@dataclass(frozen=True)
class StorylineFigure:
    """A drawn figure, its file name and its caption (sample size, source, years)."""

    name: str
    figure: Figure
    caption: str


def storyline_figures(
    clean: pd.DataFrame,
    links: pd.DataFrame,
    joined: pd.DataFrame,
    filings: pd.DataFrame,
) -> list[StorylineFigure]:
    """The four storyline figures, from the clean panel, links, joined panel and filings."""
    return [
        join_funnel(clean, links, joined, filings),
        leverage_tail_loss(joined),
        section_lengths(filings),
        going_concern_items(filings),
    ]


def write_figures(figures: list[StorylineFigure], folder: Path) -> Path:
    """Write each figure as PDF and PNG, and every caption to `captions.md`; return that path."""
    folder.mkdir(parents=True, exist_ok=True)
    for item in figures:
        item.figure.savefig(folder / f"{item.name}.pdf", metadata={"CreationDate": None})
        item.figure.savefig(folder / f"{item.name}.png", dpi=200)
    captions = folder / "captions.md"
    captions.write_text(
        "# Figure captions\n\nRegenerated with the figures; paste into the report.\n"
        + "".join(f"\n## {item.name}\n\n{item.caption}\n" for item in figures)
    )
    return captions


def draw_figures(settings: Settings) -> Path:
    """Redraw the storyline figures from the tables under `settings.data_dir`."""
    figures = storyline_figures(
        pd.read_parquet(settings.interim_dir / "kaggle_clean.parquet"),
        pd.read_parquet(settings.interim_dir / "company_links.parquet"),
        pd.read_parquet(settings.processed_dir / "panel.parquet"),
        pd.read_parquet(settings.processed_dir / "filings.parquet"),
    )
    return write_figures(figures, settings.figures_dir)


# --- 1. Join funnel ----------------------------------------------------------


def join_funnel(
    clean: pd.DataFrame,
    links: pd.DataFrame,
    joined: pd.DataFrame,
    filings: pd.DataFrame,
) -> StorylineFigure:
    """Companies and tail losers left at each step from the panel to the needed 10-K Items."""
    steps = join_steps(clean, links, joined).groupby("step", sort=False).sum()
    has_10k = filings["filing_date"].notna()
    in_corpus = filings["in_corpus"].astype(bool)
    has_items = (
        filings["has_item_1A"].fillna(False)
        & filings["has_item_7"].fillna(False)
        & (filings["has_item_8"].fillna(False) | filings["has_item_15"].fillna(False))
    ).astype(bool)

    def counted(rows: pd.Series) -> pd.Series:
        return pd.Series(
            {"companies": int(rows.sum()), "tail_losers": int(filings.loc[rows, "tail_loss"].sum())}
        )

    funnel = pd.DataFrame(
        {
            "Clean panel": steps.loc["clean panel"],
            "Linked by today's ticker alone": steps.loc["linked by ticker"],
            "+ recovered by figure matching": steps.loc["clean panel"] - steps.loc["unlinked"],
            "One row per filer (joined panel)": steps.loc["joined panel"],
            "10-K for the fiscal year found": counted(has_10k),
            "10-K text in the corpus": counted(in_corpus),
            "Items 1A, 7 and 8 or 15 present": counted(has_items),
        }
    ).T
    share = funnel / funnel.iloc[0]

    figure = _new_figure(clean, figsize=(7.5, 4.4))
    ax = figure.add_subplot()
    y = np.arange(len(funnel))
    height = 0.38
    for offset, column, color, label in [
        (-height / 2, "companies", BLUE, "All companies"),
        (
            height / 2,
            "tail_losers",
            ORANGE,
            f"Tail losers (worst {TAIL_SHARE:.0%} of next-year returns)",
        ),
    ]:
        ax.barh(y + offset, share[column], height=height - 0.04, color=color, label=label)
        for yi, value, count in zip(y, share[column], funnel[column], strict=True):
            ax.text(
                float(value) + 0.01,
                float(yi + offset),
                f"{value:.0%}  ({count:,})",
                va="center",
                fontsize=7.5,
                color=INK_SECONDARY,
            )
    ax.set_yticks(y, funnel.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of the clean panel left")
    figure.legend(loc="outside lower center", ncols=2, fontsize=8, frameon=False)
    _style(ax, grid_axis="x")
    ax.set_title(
        "Figure matching keeps the tail losers a ticker join would lose",
        loc="left",
        fontsize=10,
    )

    years = _years(clean)
    total = funnel.iloc[0]
    caption = (
        f"Companies and tail losers left at each step from the fundamentals panel to the "
        f"10-K sections the text pipeline reads, {years}, pooled over years. Steps 2 and 3 "
        f"split the linked rows: a ticker join alone keeps {share.iloc[1]['companies']:.0%} "
        f"of companies but {share.iloc[1]['tail_losers']:.0%} of tail losers; matching "
        f"reported figures to SEC XBRL data raises this to "
        f"{share.iloc[2]['companies']:.0%} and {share.iloc[2]['tail_losers']:.0%}. "
        f"n = {total['companies']:,} company-years ({total['tail_losers']:,} tail losers). "
        f"Source: {KAGGLE}; {SEC}; {CORPUS}."
    )
    return StorylineFigure("join_funnel", figure, caption)


# --- 2. Leverage against tail-loss rate ---------------------------------------


def leverage_tail_loss(joined: pd.DataFrame) -> StorylineFigure:
    """Tail-loss rate by debt-to-equity quintile, market-wide and within sectors.

    Debt-to-equity is cut into quintiles within each fiscal year, either over
    the whole market or within each sector. Negative equity (debt-to-equity
    below zero) is its own group: the ratio's order is meaningless there.
    """
    rows = joined[["fiscal_year", "sector", LEVERAGE, "tail_loss"]].dropna()
    negative = rows[rows[LEVERAGE] < 0]
    rows = rows[rows[LEVERAGE] >= 0].copy()
    rows["market_quintile"] = rows.groupby("fiscal_year")[LEVERAGE].transform(_quintile)
    rows["sector_quintile"] = rows.groupby(["fiscal_year", "sector"])[LEVERAGE].transform(_quintile)
    market = rows.groupby("market_quintile")["tail_loss"].mean()
    within = rows.groupby("sector_quintile")["tail_loss"].mean()
    sectors = rows.groupby("sector").agg(
        median_leverage=(LEVERAGE, "median"), tail_loss_rate=("tail_loss", "mean")
    )
    base_rate = pd.concat([rows, negative])["tail_loss"].mean()

    figure = _new_figure(joined, figsize=(9, 4))
    left, right = figure.subplots(1, 2, width_ratios=[1.15, 1])
    x = np.arange(1, QUINTILES + 1)
    for series, color, label in [
        (market, BLUE, "Quintiles across the market"),
        (within, ORANGE, "Quintiles within each sector"),
    ]:
        values = series.reindex(x)
        left.plot(x, values, color=color, linewidth=2, marker="o", markersize=6, label=label)
    if len(negative):
        rate = negative["tail_loss"].mean()
        left.plot(0, rate, marker="D", markersize=6, color=INK_SECONDARY)
        left.annotate(
            f"{rate:.0%}",
            (0, rate),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=INK_SECONDARY,
        )
    left.axhline(base_rate, color=INK_MUTED, linewidth=1, linestyle=":")
    left.text(
        QUINTILES + 0.45,
        base_rate,
        f"all companies\n{base_rate:.0%}",
        fontsize=7.5,
        color=INK_MUTED,
        va="center",
    )
    left.set_xticks(
        [0, *x], ["Negative\nequity", "Q1\nleast debt", "Q2", "Q3", "Q4", "Q5\nmost debt"]
    )
    left.set_xlim(-0.5, QUINTILES + 1.3)
    left.set_ylim(0, None)
    left.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    left.set_ylabel("Tail-loss rate")
    left.set_xlabel("Debt-to-equity quintile, cut each fiscal year")
    left.legend(loc="lower left", fontsize=8, frameon=False)
    left.set_title("More debt, fewer tail losses: the wrong way round", loc="left", fontsize=10)
    _style(left, grid_axis="y")

    right.scatter(
        sectors["median_leverage"],
        sectors["tail_loss_rate"],
        s=36,
        color=BLUE,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )
    for sector, row in sectors.iterrows():
        offset, align = LABEL_OFFSETS.get(str(sector), ((5, 3), "left"))
        right.annotate(
            str(sector),
            (row["median_leverage"], row["tail_loss_rate"]),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            fontsize=7,
            color=INK_SECONDARY,
        )
    right.set_ylim(0, None)
    right.set_xlim(0, None)
    right.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    right.set_xlabel("Median debt-to-equity of the sector")
    right.set_ylabel("Tail-loss rate of the sector")
    right.set_title("Why: the most indebted sectors are the safest", loc="left", fontsize=10)
    _style(right, grid_axis="both")

    market, within = market.reindex(x), within.reindex(x)
    lowest = QUINTILE_NAMES[int(np.nanargmin(market.to_numpy()))]
    caption = (
        f"Share of companies in the worst {TAIL_SHARE:.0%} of next-year returns, by "
        f"debt-to-equity quintile, {_years(joined)}, pooled over years. Left: with quintiles "
        f"cut over the whole market each year (blue), the rate goes from {market.iloc[0]:.0%} "
        f"for the least indebted fifth to {market.min():.0%} for the {lowest} and "
        f"{market.iloc[-1]:.0%} for the most indebted, the opposite of the textbook reading "
        f"that debt adds risk. With quintiles cut within each sector and year (orange), it goes "
        f"from {within.iloc[0]:.0%} to {within.iloc[-1]:.0%}. The dotted line is the rate "
        f"over all companies shown ({base_rate:.0%}). Companies "
        f"with negative equity (diamond, n = {len(negative):,}) have a tail-loss rate of "
        f"{negative['tail_loss'].mean():.0%}. Right: each sector's median debt-to-equity "
        f"against its tail-loss rate. n = {len(rows) + len(negative):,} company-years of the "
        f"joined panel with a debt-to-equity ratio. Source: {KAGGLE}."
    )
    return StorylineFigure("leverage_tail_loss", figure, caption)


def _quintile(values: pd.Series) -> pd.Series:
    """Quintile 1-5 of each value within `values`; ties are split by order."""
    order = values.rank(method="first").astype(int) - 1
    return order * QUINTILES // len(values) + 1


# --- 3. Section lengths against the context window ------------------------------


def section_lengths(filings: pd.DataFrame) -> StorylineFigure:
    """Estimated tokens of Items 1A, 7 and 8, against the LLM's context window."""
    figure = _new_figure(filings, figsize=(9, 3.4))
    axes = figure.subplots(1, len(WINDOW_ITEMS), sharex=True, sharey=True)
    bins = np.logspace(*LENGTH_RANGE, 12 * (LENGTH_RANGE[1] - LENGTH_RANGE[0]) + 1)
    summaries = []
    for ax, item in zip(axes, WINDOW_ITEMS, strict=True):
        present = filings[f"has_item_{item}"].fillna(False).astype(bool)
        tokens = filings.loc[present, f"tokens_item_{item}"].dropna().astype(float)
        over = (tokens > CONTEXT_WINDOW_TOKENS).mean() if len(tokens) else 0.0
        summaries.append(f"Item {item} {over:.1%} of {len(tokens):,}")
        ax.hist(
            tokens.clip(bins[0], bins[-1]),
            bins=bins,
            color=BLUE,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(CONTEXT_WINDOW_TOKENS, color=ORANGE, linewidth=2)
        ax.set_xscale("log")
        ax.set_title(f"Item {item}: {over:.1%} over the window", loc="left", fontsize=10)
        ax.text(
            0.02,
            0.95,
            f"n = {len(tokens):,}",
            transform=ax.transAxes,
            fontsize=7.5,
            color=INK_SECONDARY,
            va="top",
        )
        ax.set_xlabel("Estimated tokens (characters ÷ 4)")
        _style(ax, grid_axis="y")
    axes[0].set_ylabel("10-Ks")
    axes[-1].annotate(
        f"{CONTEXT_WINDOW_TOKENS // 1024}k window",
        (CONTEXT_WINDOW_TOKENS, 0.85),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=INK_SECONDARY,
    )

    caption = (
        f"Estimated length of Items 1A (risk factors), 7 (management's discussion) and 8 "
        f"(financial statements) of the joined panel's 10-Ks, {_years(filings)}, on a log "
        f"scale; the orange line is the {CONTEXT_WINDOW_TOKENS:,}-token context of the local "
        f"LLM, before the prompt and answer take their share. Share of the sections present "
        f"that are longer than the window: {'; '.join(summaries)}. Absent and stub sections "
        f"are left out, so n differs by Item; many 10-Ks put their statements in Item 15 and "
        f"leave Item 8 a stub. Lengths outside {bins[0]:,.0f} to {bins[-1]:,.0f} tokens are "
        f"drawn in the end bins. "
        f"n = {int(filings['in_corpus'].sum()):,} 10-Ks with text. Source: {CORPUS}."
    )
    return StorylineFigure("section_lengths", figure, caption)


# --- 4. Where going-concern wording appears -----------------------------------


def going_concern_items(filings: pd.DataFrame) -> StorylineFigure:
    """10-Ks whose Item contains "going concern", for each Item.

    Items mentioned in fewer than `MINOR_ITEM_SHARE` of those 10-Ks are drawn
    as one bar, which counts the 10-Ks with the words in any of them.
    """
    with_text = filings[filings["in_corpus"].astype(bool)]
    items = with_text["gc_keyword_items"].fillna("").str.split(";")
    items = items.map(lambda found: {item for item in found if item})
    any_mention = int(items.map(bool).sum())
    outside_only = int(items.map(lambda found: bool(found) and not found & STATEMENT_ITEMS).sum())
    beyond = int(items.map(lambda found: bool(found - STATEMENT_ITEMS)).sum())

    counts = pd.Series([item for found in items for item in found], dtype="str").value_counts()
    major = {str(item) for item, count in counts.items() if count >= MINOR_ITEM_SHARE * any_mention}
    order = [item for item in ITEM_ORDER if item in major]
    order += sorted(major - set(order))
    minor = set(counts.index) - major
    labels = [f"Item {item}" for item in order]
    values = [int(counts[item]) for item in order]
    colors = [ORANGE if item in STATEMENT_ITEMS else BLUE for item in order]
    if minor:
        labels.append(f"{len(minor)} other Items")
        values.append(int(items.map(lambda found: bool(found & minor)).sum()))
        colors.append(BLUE)

    figure = _new_figure(filings, figsize=(7.5, 3.8))
    ax = figure.add_subplot()
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, height=0.7)
    for yi, count in zip(y, values, strict=True):
        ax.text(
            float(count),
            float(yi),
            f"  {count:,} ({count / max(any_mention, 1):.0%})",
            va="center",
            fontsize=7.5,
            color=INK_SECONDARY,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values, default=1) * 1.2)
    ax.set_xlabel('10-Ks whose Item contains "going concern"')
    figure.legend(
        handles=[
            _swatch(ORANGE, "Items 8 and 15: financial statements and auditor's report"),
            _swatch(BLUE, "Other Items"),
        ],
        loc="outside lower center",
        ncols=2,
        fontsize=8,
        frameon=False,
    )
    ax.set_title(
        "Going-concern wording is spread across the 10-K, not only the audit opinion",
        loc="left",
        fontsize=10,
    )
    _style(ax, grid_axis="x")

    caption = (
        f'Number of 10-Ks whose Item contains the words "going concern", {_years(filings)}; '
        f"percentages are of the {any_mention:,} 10-Ks with the words anywhere, and a 10-K "
        f"counts once in each Item that has them. {beyond:,} of these 10-Ks "
        f"({beyond / max(any_mention, 1):.0%}) have the words in an Item other than 8 and 15, "
        f"where the financial statements and the auditor's report sit, and {outside_only:,} "
        f"({outside_only / max(any_mention, 1):.0%}) have them only outside those two Items. "
        f"A keyword search alone cannot tell an auditor's doubt from these other mentions. "
        f"n = {len(with_text):,} 10-Ks with text. Source: {CORPUS}."
    )
    return StorylineFigure("going_concern_items", figure, caption)


# --- Shared ------------------------------------------------------------------


def _new_figure(table: pd.DataFrame, figsize: tuple[float, float]) -> Figure:
    """An empty figure whose top line names the fiscal and return years of `table`."""
    figure = Figure(figsize=figsize, layout="constrained")
    years = _years(table)
    figure.suptitle(years[0].upper() + years[1:], x=0.01, ha="left", fontsize=8, color=INK_MUTED)
    return figure


def _years(table: pd.DataFrame) -> str:
    """'fiscal 2014–2018, returns in 2015–2019' for the years in `table`."""
    first, last = int(table["fiscal_year"].min()), int(table["fiscal_year"].max())
    return f"fiscal {first}–{last}, returns in {first + 1}–{last + 1}"


def _style(ax: Axes, grid_axis: Literal["both", "x", "y"]) -> None:
    """Recessive axes: no top or right spine, a light grid behind the marks."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _swatch(color: str, label: str) -> Patch:
    """A legend entry for bars of `color`."""
    return Patch(color=color, label=label)


def main() -> None:
    path = draw_figures(Settings.from_env())
    print(f"storyline figures: written to {path.parent}")


if __name__ == "__main__":
    main()
