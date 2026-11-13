"""What every report figure shares: its record, the files it is written to, its look.

Each figure is drawn from the pipeline's tables only, and written as PDF (for
the report) and PNG (to look at), with its caption in `captions.md`. A
caption states the sample size (n), the source, and the fiscal and return
years.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

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
SURFACE = "#f5f4f0"

QUINTILES = 5


@dataclass(frozen=True)
class ReportFigure:
    """A drawn figure, its file name and its caption (sample size, source, years)."""

    name: str
    figure: Figure
    caption: str


def write_figures(figures: list[ReportFigure], folder: Path) -> Path:
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


def new_figure(table: pd.DataFrame, figsize: tuple[float, float]) -> Figure:
    """An empty figure whose top line names the fiscal and return years of `table`."""
    figure = Figure(figsize=figsize, layout="constrained")
    years = years_label(table)
    figure.suptitle(years[0].upper() + years[1:], x=0.01, ha="left", fontsize=8, color=INK_MUTED)
    return figure


def years_label(table: pd.DataFrame) -> str:
    """'fiscal 2014–2018, returns in 2015–2019' for the years in `table`."""
    first, last = int(table["fiscal_year"].min()), int(table["fiscal_year"].max())
    if first == last:
        return f"fiscal {first}, returns in {first + 1}"
    return f"fiscal {first}–{last}, returns in {first + 1}–{last + 1}"


def style_axes(ax: Axes, grid_axis: Literal["both", "x", "y"] | None) -> None:
    """Recessive axes: no top or right spine, a light grid behind the marks."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def percent_axis(ax: Axes, axis: Literal["x", "y"] = "y") -> None:
    """Tick labels of `axis` as whole percentages of a share (0.25 -> 25%)."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(lambda value, _: f"{value:.0%}")


def quintile(values: pd.Series) -> pd.Series:
    """Quintile 1-5 of each value within `values`; ties are split by order."""
    order = values.rank(method="first").astype(int) - 1
    return order * QUINTILES // len(values) + 1


def swatch(color: str, label: str) -> Patch:
    """A legend entry for bars of `color`."""
    return Patch(color=color, label=label)
