"""The rest of the exploratory section: what the data holds before the system appears.

The data and its coverage:
- `companies_per_sector`: how the joined panel spreads over sectors.
- `missing_values`: share of fundamentals missing, by column group and sector.

Fundamentals:
- `market_cap`: most companies are small and a few are giant.
- `valuation_by_sector`: price-to-earnings and price-to-book by sector.
- `criteria_correlation`: which of the six textbook criteria measure the same thing.

The outcome:
- `return_distribution`: next-year returns per year, with each year's tail cut-off.
- `tail_loss_by_size_sector`: tail-loss rate by size quintile and by sector.

The 10-K text:
- `item1a_length`: risk-factor sections over 2014–2018.
- `filing_lag`: days from fiscal year end to 10-K, tail losers against others.
- `item1a_change`: how much of Item 1A is new or changed each year.
- `worked_example`: one company's fundamentals beside quoted 10-K passages.

These figures describe the data only; no model result appears in them.
Drawn with the conventions in `common.py`; `draw.py` writes them.
"""

import textwrap

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ai_mcdm_invest.data.panel import TAIL_SHARE, TEST_YEARS
from ai_mcdm_invest.figures.column_groups import COLUMN_GROUPS
from ai_mcdm_invest.figures.common import (
    BLUE,
    CORPUS,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    KAGGLE,
    ORANGE,
    QUINTILES,
    SEC,
    SURFACE,
    ReportFigure,
    new_figure,
    percent_axis,
    quintile,
    style_axes,
    swatch,
    years_label,
)
from ai_mcdm_invest.figures.storyline import LEVERAGE

# The six textbook ratios of the classic MCDM layer (decision 4.1), with labels.
CRITERIA = {
    "debtEquityRatio": "Debt-to-equity",
    "currentRatio": "Current ratio",
    "returnOnEquity": "Return on equity",
    "netProfitMargin": "Net margin",
    "priceEarningsRatio": "Price-to-earnings",
    "interestCoverage": "Interest coverage",
}
MARKET_CAP = "Market Cap"
PRICE_EARNINGS = "priceEarningsRatio"
PRICE_BOOK = "priceToBookRatio"

# Axis limits; values outside are drawn at the limit and counted in the caption.
MARKET_CAP_RANGE = (1e6, 1e12)
PE_RANGE = (1.0, 1000.0)
PB_RANGE = (0.1, 100.0)
RETURN_RANGE = (-100.0, 200.0)
LAG_RANGE_DAYS = 180
# Box-plot whiskers, as percentiles: skewed ratios make 1.5 times the box meaningless.
WHISKER_PERCENTILES = (5.0, 95.0)
# SEC 10-K deadlines after the fiscal year end: large accelerated, accelerated, other filers.
DEADLINES_DAYS = [60, 75, 90]
# Size bands (market cap, US dollars) commonly used to name companies.
SIZE_BANDS = {300e6: "$300M", 2e9: "$2B", 10e9: "$10B"}
# Share of the largest companies whose share of total market cap the caption gives.
TOP_SHARE = 0.10
# Item 1A change share below which a year's risk factors count as barely changed.
LITTLE_CHANGE = 0.10

# Sequential (one hue) and diverging (blue <-> red, grey midpoint) colour maps.
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#86b6ef", "#256abf", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#1c5cab", "#f0efec", "#e34948"])

ITEM_NAMES = {
    "8": "Item 8, financial statements and auditor's report",
    "15": "Item 15, financial statements and auditor's report",
    "1A": "Item 1A, risk factors",
    "3": "Item 3, legal proceedings",
}
QUOTE_WIDTH_CHARS = 105
# Ratios with equity in the denominator: their sign flips when equity is negative.
EQUITY_RATIOS = {"debtEquityRatio", "returnOnEquity"}


def exploratory_figures(
    joined: pd.DataFrame, filings: pd.DataFrame, passages: pd.DataFrame
) -> list[ReportFigure]:
    """The exploratory figures, from the joined panel, filings and worked-example passages."""
    return [
        companies_per_sector(joined),
        missing_values(joined),
        market_cap(joined),
        valuation_by_sector(joined),
        criteria_correlation(joined),
        return_distribution(joined),
        tail_loss_by_size_sector(joined),
        item1a_length(filings),
        filing_lag(filings),
        item1a_change(filings),
        worked_example(joined, passages),
    ]


# --- The data and its coverage ------------------------------------------------


def companies_per_sector(joined: pd.DataFrame) -> ReportFigure:
    """Filers per sector over all years, sorted, with company-years beside."""
    counts = joined.groupby("sector").agg(
        companies=("cik", "nunique"), company_years=("cik", "size")
    )
    counts = counts.sort_values("companies", ascending=False)
    total = joined["cik"].nunique()

    figure = new_figure(joined, figsize=(7, 3.8))
    ax = figure.add_subplot()
    y = np.arange(len(counts))
    ax.barh(y, counts["companies"], color=BLUE, height=0.7)
    for yi, (companies, company_years) in zip(y, counts.to_numpy(), strict=True):
        ax.text(
            float(companies),
            float(yi),
            f"  {companies:,} ({companies / total:.0%}); {company_years:,} company-years",
            va="center",
            fontsize=7.5,
            color=INK_SECONDARY,
        )
    ax.set_yticks(y, counts.index)
    ax.invert_yaxis()
    ax.set_xlim(0, counts["companies"].max() * 1.6)
    ax.set_xlabel("Companies (SEC filers) in the joined panel")
    ax.set_title("Companies per sector", loc="left", fontsize=10)
    style_axes(ax, grid_axis="x")

    largest, smallest = counts.index[0], counts.index[-1]
    caption = (
        f"Companies (distinct SEC filers) per sector in the joined panel, {years_label(joined)}, "
        f"with their share of all companies and their company-years. {largest} is the largest "
        f"sector ({counts['companies'].iloc[0] / total:.0%} of companies) and {smallest} the "
        f"smallest ({counts['companies'].iloc[-1] / total:.0%}), so pooled figures lean on "
        f"the large sectors. A company that changed sector counts in each. "
        f"n = {total:,} companies, {len(joined):,} company-years. Source: {KAGGLE}; {SEC}."
    )
    return ReportFigure("companies_per_sector", figure, caption)


def missing_values(joined: pd.DataFrame) -> ReportFigure:
    """Share of fundamentals missing, by column group (rows) and sector (columns)."""
    groups = {
        name: [column for column in columns if column in joined.columns]
        for name, columns in COLUMN_GROUPS.items()
    }
    groups = {name: columns for name, columns in groups.items() if columns}
    sectors = sorted(joined["sector"].unique())
    share = pd.DataFrame(
        {
            name: [
                *(
                    joined.loc[joined["sector"] == sector, columns].isna().to_numpy().mean()
                    for sector in sectors
                ),
                joined[columns].isna().to_numpy().mean(),
            ]
            for name, columns in groups.items()
        },
        index=[*sectors, "All sectors"],
    ).T

    figure = new_figure(joined, figsize=(9, 3.6))
    ax = figure.add_subplot()
    top = max(float(share.to_numpy().max()), 0.01)
    image = ax.imshow(share.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=top, aspect="auto")
    for (row, column), value in np.ndenumerate(share.to_numpy()):
        ax.text(
            column,
            row,
            f"{value:.0%}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if value > top * 0.55 else INK,
        )
    ax.set_xticks(range(share.shape[1]), share.columns, rotation=30, ha="right")
    ax.set_yticks(range(share.shape[0]), [f"{name} ({len(groups[name])})" for name in share.index])
    ax.axvline(len(sectors) - 0.5, color="white", linewidth=3)
    colorbar = figure.colorbar(image, ax=ax, shrink=0.8, pad=0.01)
    percent_axis(colorbar.ax)
    colorbar.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    ax.set_title("Share of values missing, by column group and sector", loc="left", fontsize=10)
    style_axes(ax, grid_axis=None)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)

    overall = share["All sectors"]
    caption = (
        f"Share of fundamental values missing in the joined panel, {years_label(joined)}, by "
        f"column group (number of columns in brackets, grouped by the section of the source "
        f"they come from) and sector; the last column is all sectors together. Missing shares "
        f"range from {overall.min():.0%} ({overall.idxmin()}) to {overall.max():.0%} "
        f"({overall.idxmax()}). Rows with missing fundamentals are kept; each later stage "
        f"says how it treats a gap. n = {len(joined):,} company-years. Source: {KAGGLE}."
    )
    return ReportFigure("missing_values", figure, caption)


# --- Fundamentals --------------------------------------------------------------


def market_cap(joined: pd.DataFrame) -> ReportFigure:
    """Histogram of market cap on a log scale, with size bands and the median."""
    caps = joined.loc[joined[MARKET_CAP] > 0, ["fiscal_year", MARKET_CAP]]
    values = caps[MARKET_CAP].astype(float)
    bins = np.logspace(np.log10(MARKET_CAP_RANGE[0]), np.log10(MARKET_CAP_RANGE[1]), 49)
    top_shares = caps.groupby("fiscal_year")[MARKET_CAP].apply(_top_share)
    median = float(values.median())

    figure = new_figure(joined, figsize=(7.5, 3.6))
    ax = figure.add_subplot()
    ax.hist(
        values.clip(bins[0], bins[-1]),
        bins=bins.tolist(),
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xscale("log")
    for cap, label in SIZE_BANDS.items():
        ax.axvline(cap, color=INK_MUTED, linewidth=1, linestyle=":")
        ax.annotate(
            label,
            (cap, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -10),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_MUTED,
        )
    ax.axvline(median, color=ORANGE, linewidth=2)
    ax.annotate(
        f"median {_dollars(median)}",
        (median, 0.82),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=INK_SECONDARY,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    ax.xaxis.set_major_formatter(lambda value, _: _dollars(value))
    ax.set_xlabel("Market cap at the fiscal year end (US dollars, log scale)")
    ax.set_ylabel("Company-years")
    ax.set_title("Most companies are small; a few are giant", loc="left", fontsize=10)
    style_axes(ax, grid_axis="y")

    outside = int(((values < bins[0]) | (values > bins[-1])).sum())
    caption = (
        f"Market cap of the joined panel's companies, {years_label(joined)}, on a log scale, "
        f"pooled over years. The median is {_dollars(median)}; "
        f"{(values < 2e9).mean():.0%} of company-years are below $2B, and in each year the "
        f"largest {TOP_SHARE:.0%} of companies hold {top_shares.min():.0%}–{top_shares.max():.0%} "
        f"of the total market cap. Dotted lines mark the usual small-, mid- and large-cap "
        f"bounds. {outside:,} values outside {_dollars(bins[0])} to {_dollars(bins[-1])} are "
        f"drawn in the end bins; {len(joined) - len(caps):,} rows without a positive market "
        f"cap are left out. n = {len(caps):,} company-years. Source: {KAGGLE}."
    )
    return ReportFigure("market_cap", figure, caption)


def _top_share(caps: pd.Series) -> float:
    """Share of the sum of `caps` held by its largest `TOP_SHARE` values."""
    largest = caps.nlargest(max(1, round(len(caps) * TOP_SHARE)))
    return float(largest.sum() / caps.sum())


def _dollars(value: float) -> str:
    """'$2.3B', '$450M', '$1T'."""
    for size, unit in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
        if value >= size:
            return f"${value / size:.3g}{unit}"
    return f"${value:,.0f}"


def valuation_by_sector(joined: pd.DataFrame) -> ReportFigure:
    """Price-to-earnings and price-to-book by sector, box plots on log scales."""
    sectors = (
        joined[joined[PRICE_EARNINGS] > 0]
        .groupby("sector")[PRICE_EARNINGS]
        .median()
        .sort_values()
        .index.tolist()
    )
    figure = new_figure(joined, figsize=(9, 4.2))
    axes = figure.subplots(1, 2, sharey=True)
    summaries = []
    for ax, column, limits, label in [
        (axes[0], PRICE_EARNINGS, PE_RANGE, "Price-to-earnings"),
        (axes[1], PRICE_BOOK, PB_RANGE, "Price-to-book"),
    ]:
        positive = joined[joined[column] > 0]
        data = [
            positive.loc[positive["sector"] == sector, column].clip(*limits).to_numpy()
            for sector in sectors
        ]
        box = ax.boxplot(
            data,
            orientation="horizontal",
            tick_labels=sectors,
            whis=WHISKER_PERCENTILES,
            showfliers=False,
            widths=0.6,
            patch_artist=True,
            medianprops={"color": ORANGE, "linewidth": 2},
            whiskerprops={"color": INK_MUTED},
            capprops={"color": INK_MUTED},
        )
        for patch in box["boxes"]:
            patch.set(facecolor=BLUE, edgecolor=BLUE, alpha=0.35)
        ax.set_xscale("log")
        ax.set_xlim(*limits)
        ax.set_xlabel(f"{label} (log scale, positive values)")
        ax.set_title(label, loc="left", fontsize=10)
        style_axes(ax, grid_axis="x")
        outside = int(((positive[column] < limits[0]) | (positive[column] > limits[1])).sum())
        summaries.append(
            f"{label}: {len(positive):,} positive values, median {positive[column].median():.1f}, "
            f"{outside:,} outside {limits[0]:g}–{limits[1]:g} drawn at the limit"
        )
    axes[0].invert_yaxis()

    losses = joined["Net Income"] < 0 if "Net Income" in joined.columns else pd.Series(False)
    not_positive = (joined[PRICE_EARNINGS] <= 0).mean()
    caption = (
        f"Price-to-earnings and price-to-book by sector, {years_label(joined)}, pooled over "
        f"years, sectors sorted by median price-to-earnings. Boxes span the middle half, the "
        f"orange line is the median and whiskers reach the 5th and 95th percentiles; points "
        f"beyond are not drawn. {'; '.join(summaries)}. Price-to-earnings is not meaningful "
        f"for a loss: "
        f"{losses.mean():.0%} of company-years report negative net income, and "
        f"{not_positive:.0%} have a price-to-earnings of zero or below, left out here. "
        f"n = {len(joined):,} company-years. Source: {KAGGLE}."
    )
    return ReportFigure("valuation_by_sector", figure, caption)


def criteria_correlation(joined: pd.DataFrame) -> ReportFigure:
    """Rank correlation of the six textbook criteria, on within-year percentiles."""
    complete = joined.dropna(subset=list(CRITERIA))
    percentiles = complete[list(CRITERIA)].groupby(complete["fiscal_year"]).rank(pct=True)
    correlation = percentiles.corr().rename(index=CRITERIA, columns=CRITERIA)
    values = correlation.to_numpy()
    labels = list(correlation.columns)
    # Each pair once: the matrix below its diagonal, without the diagonal's ones.
    pairs = values[1:, :-1]
    below = np.tril(np.ones_like(pairs, dtype=bool))

    figure = new_figure(joined, figsize=(6.2, 4.4))
    ax = figure.add_subplot()
    image = ax.imshow(np.where(below, pairs, np.nan), cmap=DIVERGING, vmin=-1, vmax=1)
    for (row, column), value in np.ndenumerate(pairs):
        if below[row, column]:
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(float(value)) > 0.6 else INK,
            )
    ax.set_xticks(range(len(labels) - 1), labels[:-1], rotation=30, ha="right")
    ax.set_yticks(range(len(labels) - 1), labels[1:])
    colorbar = figure.colorbar(image, ax=ax, shrink=0.75, pad=0.02)
    colorbar.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Rank correlation", fontsize=8, color=INK_SECONDARY)
    ax.set_title("Which criteria measure the same thing", loc="left", fontsize=10)
    style_axes(ax, grid_axis=None)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)

    lower = list(zip(*np.tril_indices(len(labels), -1), strict=True))
    row, column = max(lower, key=lambda pair: abs(float(values[pair])))
    strongest = (labels[row], labels[column])
    caption = (
        f"Rank correlation between the six textbook criteria of the classic MCDM layer, "
        f"{years_label(joined)}: each criterion is turned into its percentile within the fiscal "
        f"year, as the ranking scores it, and the percentiles are correlated (Pearson) over all "
        f"years; no criterion direction is applied. The strongest pair is "
        f"{strongest[0].lower()} and {strongest[1].lower()} ({values[row, column]:+.2f}): "
        f"weighting both partly counts the same signal twice. Only company-years with all six "
        f"criteria are used. "
        f"n = {len(complete):,} company-years. Source: {KAGGLE}."
    )
    return ReportFigure("criteria_correlation", figure, caption)


# --- The outcome ---------------------------------------------------------------


def return_distribution(joined: pd.DataFrame) -> ReportFigure:
    """Next-year returns per fiscal year, with each year's tail-loss cut-off."""
    years = sorted(joined["fiscal_year"].unique())
    bins = np.arange(RETURN_RANGE[0], RETURN_RANGE[1] + 5, 5)
    figure = new_figure(joined, figsize=(7.5, 1.3 * len(years) + 1))
    axes = np.atleast_1d(figure.subplots(len(years), 1, sharex=True))
    cutoffs, medians = {}, {}
    for ax, year in zip(axes, years, strict=True):
        rows = joined[joined["fiscal_year"] == year]
        returns = rows["next_year_return"].clip(*RETURN_RANGE)
        cutoff = float(rows.loc[rows["tail_loss"], "next_year_return"].max())
        cutoffs[year], medians[year] = cutoff, float(rows["next_year_return"].median())
        tail = rows["tail_loss"].astype(bool)
        ax.hist(
            [returns[tail], returns[~tail]],
            bins=bins.tolist(),
            stacked=True,
            color=[ORANGE, BLUE],
            linewidth=0,
        )
        ax.axvline(cutoff, color=INK, linewidth=1)
        ax.axvline(0, color=INK_MUTED, linewidth=1, linestyle=":")
        ax.text(
            0.99,
            0.85,
            f"Fiscal {year}, returns in {year + 1}: cut-off {cutoff:+.0f}%, "
            f"median {medians[year]:+.0f}%, n = {len(rows):,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=INK_SECONDARY,
        )
        ax.set_yticks([])
        style_axes(ax, grid_axis=None)
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("Next-year price change (%)")
    axes[-1].xaxis.set_major_formatter(lambda value, _: f"{value:+.0f}%")
    axes[0].set_title(
        "Each year's worst fifth is cut within the year: the market's direction differs",
        loc="left",
        fontsize=10,
    )
    figure.legend(
        handles=[
            swatch(ORANGE, f"Tail losers (worst {TAIL_SHARE:.0%} of the year)"),
            swatch(BLUE, "Other companies"),
        ],
        loc="outside lower center",
        ncols=2,
        fontsize=8,
        frameon=False,
    )

    clipped = int(
        (
            (joined["next_year_return"] < RETURN_RANGE[0])
            | (joined["next_year_return"] > RETURN_RANGE[1])
        ).sum()
    )
    cut_text = ", ".join(f"{year} {cutoffs[year]:+.0f}%" for year in years)
    median_text = ", ".join(f"{year} {medians[year]:+.0f}%" for year in years)
    caption = (
        f"Next-year price change of the joined panel's companies, one panel per fiscal year "
        f"({years_label(joined)}). Orange bars are the year's tail losers, the worst "
        f"{TAIL_SHARE:.0%} of its returns; the black line is the cut-off, which moves with the "
        f"market (by fiscal year: {cut_text}; medians {median_text}). A cut-off pooled over "
        f"years would label a falling year's ordinary companies as tail losers. {clipped:,} "
        f"returns above {RETURN_RANGE[1]:+.0f}% are drawn in the last bin. "
        f"n = {len(joined):,} company-years. Source: {KAGGLE}."
    )
    return ReportFigure("return_distribution", figure, caption)


def tail_loss_by_size_sector(joined: pd.DataFrame) -> ReportFigure:
    """Tail-loss rate by market-cap quintile (cut each year) and by sector."""
    sized = joined[joined[MARKET_CAP] > 0].copy()
    sized["size_quintile"] = sized.groupby("fiscal_year")[MARKET_CAP].transform(quintile)
    by_size = sized.groupby("size_quintile")["tail_loss"].mean().reindex(range(1, QUINTILES + 1))
    by_sector = joined.groupby("sector").agg(mean=("tail_loss", "mean")).sort_values("mean")
    base_rate = float(joined["tail_loss"].mean())

    figure = new_figure(joined, figsize=(9, 3.8))
    left, right = figure.subplots(1, 2, width_ratios=[1, 1.3])
    x = np.arange(1, QUINTILES + 1)
    left.bar(x, by_size.to_numpy(), color=BLUE, width=0.65)
    for xi, rate in zip(x, by_size.to_numpy(), strict=True):
        left.text(
            xi, rate, f"{rate:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY
        )
    left.set_xticks(x, ["Q1\nsmallest", "Q2", "Q3", "Q4", "Q5\nlargest"])
    left.set_xlabel("Market-cap quintile, cut each fiscal year")
    left.set_ylabel("Tail-loss rate")
    left.set_title("Small companies fall hardest", loc="left", fontsize=10)

    y = np.arange(len(by_sector))
    right.barh(y, by_sector["mean"], color=BLUE, height=0.7)
    for yi, rate in zip(y, by_sector["mean"], strict=True):
        right.text(rate, yi, f" {rate:.0%}", va="center", fontsize=7.5, color=INK_SECONDARY)
    right.set_yticks(y, by_sector.index)
    right.set_xlabel("Tail-loss rate")
    right.set_title("Tail-loss rate by sector", loc="left", fontsize=10)
    percent_axis(right, "x")
    left.axhline(base_rate, color=INK_MUTED, linewidth=1, linestyle=":")
    right.axvline(base_rate, color=INK_MUTED, linewidth=1, linestyle=":")
    percent_axis(left)
    style_axes(left, grid_axis="y")
    style_axes(right, grid_axis="x")
    right.text(
        base_rate,
        len(by_sector) - 0.4,
        f" all companies {base_rate:.0%}",
        fontsize=7.5,
        color=INK_MUTED,
        va="bottom",
    )

    caption = (
        f"Share of companies in the worst {TAIL_SHARE:.0%} of next-year returns, "
        f"{years_label(joined)}, pooled over years. Left: by market-cap quintile, cut within "
        f"each fiscal year; the rate goes from {by_size.iloc[0]:.0%} for the smallest fifth "
        f"to {by_size.iloc[-1]:.0%} for the largest, which motivates the size baseline. "
        f"Right: by sector, from {by_sector['mean'].iloc[0]:.0%} ({by_sector.index[0]}) to "
        f"{by_sector['mean'].iloc[-1]:.0%} ({by_sector.index[-1]}). The dotted line is the "
        f"rate over all companies ({base_rate:.0%}). n = {len(joined):,} company-years "
        f"({len(sized):,} with a positive market cap on the left). Source: {KAGGLE}."
    )
    return ReportFigure("tail_loss_by_size_sector", figure, caption)


# --- The 10-K text ---------------------------------------------------------------


def item1a_length(filings: pd.DataFrame) -> ReportFigure:
    """Estimated tokens of Item 1A per fiscal year, all filers and those present every year."""
    present = filings[filings["has_item_1A"].fillna(False).astype(bool)]
    lengths = present[["cik", "fiscal_year", "tokens_item_1A"]].dropna()
    lengths = lengths.astype({"tokens_item_1A": float})
    years = sorted(filings["fiscal_year"].unique())
    every_year = lengths.groupby("cik")["fiscal_year"].nunique() == len(years)
    balanced = lengths[lengths["cik"].isin(every_year[every_year].index)]
    medians = lengths.groupby("fiscal_year")["tokens_item_1A"].median().reindex(years)
    balanced_medians = balanced.groupby("fiscal_year")["tokens_item_1A"].median().reindex(years)

    figure = new_figure(filings, figsize=(7.5, 3.8))
    ax = figure.add_subplot()
    data = [lengths.loc[lengths["fiscal_year"] == year, "tokens_item_1A"] for year in years]
    box = ax.boxplot(
        [values.to_numpy() if len(values) else [np.nan] for values in data],
        positions=range(len(years)),
        tick_labels=[
            f"{year}\n(n = {len(values):,})" for year, values in zip(years, data, strict=True)
        ],
        whis=WHISKER_PERCENTILES,
        showfliers=False,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": BLUE, "linewidth": 2},
        whiskerprops={"color": INK_MUTED},
        capprops={"color": INK_MUTED},
    )
    for patch in box["boxes"]:
        patch.set(facecolor=BLUE, edgecolor=BLUE, alpha=0.25)
    ax.plot(
        range(len(years)),
        balanced_medians.to_numpy(),
        color=ORANGE,
        linewidth=2,
        marker="o",
        markersize=6,
        label=f"Median of the {balanced['cik'].nunique():,} filers present every year",
    )
    ax.plot([], [], color=BLUE, linewidth=2, label="Median of all filers (box: middle half)")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("Item 1A, estimated tokens")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Risk-factor sections grow year after year", loc="left", fontsize=10)
    style_axes(ax, grid_axis="y")

    first, last = years[0], years[-1]
    caption = (
        f"Estimated length of Item 1A (risk factors, characters ÷ 4) of the joined panel's "
        f"10-Ks, by fiscal year ({years_label(filings)}). Boxes span the middle half and "
        f"whiskers reach the 5th and 95th percentiles; sections beyond are not drawn. "
        f"The median grows "
        f"from {medians[first]:,.0f} tokens in fiscal {first} to {medians[last]:,.0f} in "
        f"fiscal {last}; for the {balanced['cik'].nunique():,} filers with an Item 1A in every "
        f"year (orange), from {balanced_medians[first]:,.0f} to {balanced_medians[last]:,.0f}, "
        f"so the growth is not only a change in who files. Absent and stub sections are left "
        f"out. n = {len(lengths):,} 10-Ks. Source: {CORPUS}."
    )
    return ReportFigure("item1a_length", figure, caption)


def filing_lag(filings: pd.DataFrame) -> ReportFigure:
    """Days from fiscal year end to the 10-K, tail losers against others, with deadlines."""
    development = ~filings["fiscal_year"].isin(TEST_YEARS)
    filed = filings[filings["days_after_fy_end"].notna() & development]
    lag = filed["days_after_fy_end"].astype(float)
    tail = filed["tail_loss"].astype(bool)
    bins = np.arange(0, LAG_RANGE_DAYS + 5, 5)

    figure = new_figure(filed, figsize=(7.5, 3.6))
    ax = figure.add_subplot()
    summaries = []
    for rows, color, label in [(~tail, BLUE, "Other companies"), (tail, ORANGE, "Tail losers")]:
        values = lag[rows]
        weights = np.full(len(values), 1 / max(len(values), 1))
        notices = filed.loc[rows, "nt_10k"].fillna(False).astype(bool).mean()
        ax.hist(
            values.clip(0, LAG_RANGE_DAYS),
            bins=bins.tolist(),
            weights=weights,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{label} (n = {len(values):,}; {notices:.1%} filed a late-filing notice)",
        )
        summaries.append(
            f"{label.lower()}: median {values.median():.0f} days, "
            f"{(values > max(DEADLINES_DAYS)).mean():.0%} after {max(DEADLINES_DAYS)} days, "
            f"{notices:.1%} with a late-filing notice"
        )
    for days in DEADLINES_DAYS:
        ax.axvline(days, color=INK_MUTED, linewidth=1, linestyle=":")
        ax.annotate(
            f"{days} d",
            (days, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -4),
            textcoords="offset points",
            va="top",
            fontsize=7,
            color=INK_MUTED,
        )
    ax.set_xlim(0, LAG_RANGE_DAYS)
    percent_axis(ax)
    ax.set_xlabel("Days from fiscal year end to the 10-K filing")
    ax.set_ylabel("Share of the group's 10-Ks")
    ax.legend(loc="center right", fontsize=8, frameon=False)
    ax.set_title("Tail losers file later", loc="left", fontsize=10)
    style_axes(ax, grid_axis="y")

    caption = (
        f"Days from the fiscal year end to the 10-K filing, {years_label(filed)}, as a share "
        f"of each group's 10-Ks, for tail losers (orange) and other companies (blue): "
        f"{'; '.join(summaries)}. Dotted lines are the SEC deadlines: 60 days for large "
        f"accelerated filers, 75 for accelerated filers and 90 for the others. Lags above "
        f"{LAG_RANGE_DAYS} days are drawn in the last bin. The test years (fiscal "
        f"{min(TEST_YEARS)}–{max(TEST_YEARS)}) are left out: filing features are compared with "
        f"the outcome on development years only (decision 5.1). n = {len(filed):,} 10-Ks found "
        f"for the fiscal year. Source: {SEC}; {KAGGLE}."
    )
    return ReportFigure("filing_lag", figure, caption)


def item1a_change(filings: pd.DataFrame) -> ReportFigure:
    """Distribution of the share of Item 1A paragraphs new or changed since last year."""
    rows = filings[filings["item1a_changed_share"].notna()]
    change = rows["item1a_changed_share"].astype(float)
    # Tail losers are compared with the others on the development years only (decision 5.1).
    development = rows[~rows["fiscal_year"].isin(TEST_YEARS)]
    tail = development["tail_loss"].astype(bool)
    development_change = development["item1a_changed_share"].astype(float)
    bins = np.linspace(0, 1, 21)

    figure = new_figure(filings, figsize=(7.5, 3.6))
    ax = figure.add_subplot()
    ax.hist(change, bins=bins.tolist(), color=BLUE, edgecolor="white", linewidth=0.5)
    median = float(change.median())
    ax.axvline(median, color=ORANGE, linewidth=2)
    ax.annotate(
        f"median {median:.0%}",
        (median, 0.9),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=INK_SECONDARY,
    )
    percent_axis(ax, "x")
    ax.set_xlabel("Share of Item 1A paragraphs new or changed since the prior year's 10-K")
    ax.set_ylabel("10-Ks")
    ax.set_title(
        "Most of a risk-factor section carries over from the year before",
        loc="left",
        fontsize=10,
    )
    style_axes(ax, grid_axis="y")

    caption = (
        f"Share of each 10-K's Item 1A paragraphs that are new or substantially changed "
        f"since the same filer's prior-year Item 1A (a paragraph is unchanged when at least "
        f"half its sentences appear word for word), {years_label(filings)}. The median is "
        f"{median:.0%}, and {(change < LITTLE_CHANGE).mean():.0%} of 10-Ks change less than "
        f"{LITTLE_CHANGE:.0%} of their paragraphs. On {years_label(development)} (the "
        f"development years), tail losers change a median "
        f"{development_change[tail].median():.0%} against "
        f"{development_change[~tail].median():.0%} for other companies. 10-Ks without a "
        f"prior-year Item 1A are left out. "
        f"n = {len(rows):,} 10-Ks. Source: {CORPUS}."
    )
    return ReportFigure("item1a_change", figure, caption)


def worked_example(joined: pd.DataFrame, passages: pd.DataFrame) -> ReportFigure:
    """One company's six criteria (with their rank that year) beside quoted passages."""
    ticker = str(passages["ticker"].iloc[0])
    fiscal_year = int(passages["fiscal_year"].iloc[0])
    cik = int(passages["cik"].iloc[0])
    year = joined[joined["fiscal_year"] == fiscal_year]
    example = year[year["cik"] == cik]
    row = example.iloc[0]

    negative_equity = bool(row[LEVERAGE] < 0)
    lines = []
    for column, label in CRITERIA.items():
        value = row[column]
        if pd.isna(value):
            lines.append((label, "missing", ""))
            continue
        mark = " †" if negative_equity and column in EQUITY_RATIOS else ""
        rank = (year[column].dropna() < value).mean()
        lines.append((label + mark, _ratio(column, float(value)), f"{rank:.0%}"))
    lines += [
        ("Market cap", _dollars(float(row[MARKET_CAP])), ""),
        (f"Return in {fiscal_year + 1}", f"{row['next_year_return']:+.0f}%", ""),
        ("Tail loser", "yes" if row["tail_loss"] else "no", ""),
    ]
    quotes = [
        (ITEM_NAMES.get(str(item), f"Item {item}"), textwrap.fill(f"“{text}”", QUOTE_WIDTH_CHARS))
        for item, text in zip(passages["item"], passages["passage"], strict=True)
    ]

    # Laid out in inches from the top, so the card is as tall as its content.
    title, row_step, line_step, gap = 0.75, 0.3, 0.155, 0.4
    left_height = title + 0.3 + row_step * len(lines) + 0.5
    right_height = title + sum(gap + line_step * (q.count("\n") + 1) for _, q in quotes) + 0.4
    height = max(left_height, right_height)

    def top(inches: float) -> float:
        return 1 - inches / height

    figure = new_figure(example, figsize=(10, height))
    figure.set_layout_engine("none")
    left, values_x, rank_x, right = 0.02, 0.21, 0.3, 0.34
    figure.text(left, top(title), f"{ticker} in the numbers", fontsize=10, color=INK, weight="bold")
    figure.text(left, top(title + 0.2), str(row["sector"]), fontsize=8, color=INK_MUTED)
    figure.text(values_x, top(title + 0.45), "Value", fontsize=7.5, color=INK_MUTED, ha="right")
    figure.text(
        rank_x, top(title + 0.45), "Rank in year", fontsize=7.5, color=INK_MUTED, ha="right"
    )
    for index, (label, value, rank) in enumerate(lines):
        y = top(title + 0.75 + index * row_step)
        figure.text(left, y, label, fontsize=8.5, color=INK_SECONDARY)
        figure.text(values_x, y, value, fontsize=8.5, color=INK, ha="right")
        figure.text(rank_x, y, rank, fontsize=8.5, color=INK_SECONDARY, ha="right")
    if negative_equity:
        figure.text(
            left,
            top(title + 0.75 + len(lines) * row_step),
            "† negative equity: the ratio's sign, and so its rank, is reversed",
            fontsize=7,
            color=INK_MUTED,
        )

    figure.text(right, top(title), f"{ticker} in its 10-K", fontsize=10, color=INK, weight="bold")
    y = title + 0.3
    for name, quote in quotes:
        figure.text(right + 0.005, top(y), name, fontsize=8, color=ORANGE)
        figure.text(
            right + 0.005,
            top(y + 0.12),
            quote,
            fontsize=7.5,
            color=INK,
            va="top",
            linespacing=1.35,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 4},
        )
        y += gap + line_step * (quote.count("\n") + 1)

    named = ", ".join(f"Item {item}" for item in passages["item"])
    reporting = [int(year[column].notna().sum()) for column in CRITERIA]
    sign_note = (
        "Its equity is negative, so debt-to-equity and return on equity have their sign "
        "reversed (†). "
        if negative_equity
        else ""
    )
    caption = (
        f"{ticker} ({row['sector']}), {years_label(example)}: its six textbook criteria with "
        f"their rank among the joined panel's companies that year (the share with a lower "
        f"value; no criterion direction is applied, so a high rank is not a good score), "
        f"beside passages quoted from its 10-K ({named}). The numbers place a company; the "
        f"text says why it may fall: here the "
        f"{'' if row['tail_loss'] else 'non-'}tail loser's next-year return was "
        f"{row['next_year_return']:+.0f}%. "
        f"{sign_note}"
        f"Long passages are cut at a sentence end, or at a word end when their first sentence "
        f"is too long ([…]). The company is chosen by hand from the years before the test "
        f"years. Each rank counts the company-years of fiscal {fiscal_year} that report the "
        f"criterion: n = {min(reporting):,}–{max(reporting):,}. "
        f"Source: {KAGGLE}; {CORPUS}."
    )
    return ReportFigure("worked_example", figure, caption)


def _ratio(column: str, value: float) -> str:
    """A criterion value as the report prints it: margins and returns in percent."""
    if column in {"returnOnEquity", "netProfitMargin"}:
        return f"{value:.0%}"
    return f"{value:,.2f}"
