"""Where the pipeline reads raw data from and writes derived data to.

Stages never build these paths themselves; they receive tables. Only the entry
points read `Settings`: the rebuild (`pipeline.py`), the SEC download
(`data/sec_download.py`), the manifest recorder (`data/raw_manifest.py`) and
the slow acceptance tests on real data.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = Path.home() / "maids" / "data"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "raw_manifest.json"


@dataclass(frozen=True)
class Settings:
    """`raw_dir` is read only and lives outside the repository.

    `data_dir` holds every derived table and is ignored by git.
    `manifest_path` records the exact raw files the study is built on.
    """

    raw_dir: Path = DEFAULT_RAW_DIR
    data_dir: Path = DEFAULT_DATA_DIR
    manifest_path: Path = DEFAULT_MANIFEST_PATH

    @classmethod
    def from_env(cls) -> Self:
        """Read `AI_MCDM_RAW_DIR`, `AI_MCDM_DATA_DIR` and `AI_MCDM_MANIFEST`, else defaults."""
        return cls(
            raw_dir=Path(os.environ.get("AI_MCDM_RAW_DIR", DEFAULT_RAW_DIR)).expanduser(),
            data_dir=Path(os.environ.get("AI_MCDM_DATA_DIR", DEFAULT_DATA_DIR)).expanduser(),
            manifest_path=Path(
                os.environ.get("AI_MCDM_MANIFEST", DEFAULT_MANIFEST_PATH)
            ).expanduser(),
        )

    @property
    def raw_panel_dir(self) -> Path:
        """The yearly fundamentals files (`<year>_Financial_Data.csv`)."""
        return self.raw_dir / "2014 2018"

    @property
    def raw_sec_dir(self) -> Path:
        """SEC files fetched by `make fetch`: the ticker list and XBRL frames."""
        return self.raw_dir / "sec"

    @property
    def ticker_list_path(self) -> Path:
        """The SEC's current ticker list (`company_tickers.json`)."""
        return self.raw_sec_dir / "company_tickers.json"

    @property
    def raw_frames_dir(self) -> Path:
        """XBRL frames (`<tag>_<period>.json`), one figure for every filer."""
        return self.raw_sec_dir / "frames"

    @property
    def submissions_path(self) -> Path:
        """The SEC's bulk submissions file: every filer's filing history."""
        return self.raw_sec_dir / "submissions.zip"

    @property
    def raw_corpus_dir(self) -> Path:
        """EDGAR-CORPUS: 10-Ks split by Item, one `<year>_<split>.jsonl` per label year."""
        return self.raw_dir / "edgar_corpus"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"
