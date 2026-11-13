"""Draw every report figure from the pipeline's tables: the storyline, then the rest.

Run with `make figures` to redraw from the tables on disk (no raw data
needed), or as the last step of `make all`.
"""

from pathlib import Path

import pandas as pd

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.figures.common import ReportFigure, write_figures
from ai_mcdm_invest.figures.exploratory import exploratory_figures
from ai_mcdm_invest.figures.storyline import storyline_figures


def report_figures(
    clean: pd.DataFrame,
    links: pd.DataFrame,
    joined: pd.DataFrame,
    filings: pd.DataFrame,
    passages: pd.DataFrame,
) -> list[ReportFigure]:
    """The storyline figures, then the exploratory ones, in the report's order."""
    return storyline_figures(clean, links, joined, filings) + exploratory_figures(
        joined, filings, passages
    )


def draw_figures(settings: Settings) -> Path:
    """Redraw every figure from the tables under `settings.data_dir`; return `captions.md`."""
    figures = report_figures(
        pd.read_parquet(settings.interim_dir / "kaggle_clean.parquet"),
        pd.read_parquet(settings.interim_dir / "company_links.parquet"),
        pd.read_parquet(settings.processed_dir / "panel.parquet"),
        pd.read_parquet(settings.processed_dir / "filings.parquet"),
        pd.read_parquet(settings.processed_dir / "worked_example.parquet"),
    )
    return write_figures(figures, settings.figures_dir)


def main() -> None:
    path = draw_figures(Settings.from_env())
    print(f"figures: written to {path.parent}")


if __name__ == "__main__":
    main()
