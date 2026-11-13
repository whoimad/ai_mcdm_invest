"""Download the raw files the rebuild reads besides the panel. Run with `make fetch`.

This is the only code that touches the network. It adds files under
`<raw_dir>/sec/` and `<raw_dir>/edgar_corpus/` and never rewrites one that is
already there, so the rebuild (`make all`) stays offline and repeatable.

Files:
- `sec/company_tickers.json`: the SEC's current ticker list.
- `sec/frames/<tag>_<period>.json`: one XBRL "frame" per tag and calendar
  period, holding that figure for every filer that reported it.
- `sec/submissions.zip`: every filer's filing history (forms, filing and
  report dates), about 1.6 GB.
- `edgar_corpus/<year>_<split>.jsonl`: EDGAR-CORPUS 10-Ks split by Item, from
  Hugging Face, about 1.8 GB a year.
"""

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.panel import FISCAL_YEARS

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FRAME_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json"
SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
CORPUS_URL = (
    "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/{year}/{split}.jsonl"
)

ASSETS_TAG = "Assets"
NET_INCOME_TAG = "NetIncomeLoss"
# Companies report revenue under different tags, and the tags changed in 2018.
REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "RevenuesNetOfInterestExpense",
]
# A frame holds each filer's period under the calendar year it overlaps most,
# so a fiscal year ending in January 2014 (fiscal 2014) sits in the CY2013
# frame. Hence one extra year before the panel's first.
FRAME_YEARS = range(FISCAL_YEARS.start - 1, FISCAL_YEARS.stop)

# Corpus labels are usually the fiscal year covered but can be a year off
# either way, and fiscal 2014 needs the prior year's 10-K (periods ending in
# 2013) for the Item 1A comparison. Hence labels 2012 to 2019.
CORPUS_YEARS = range(FISCAL_YEARS.start - 2, FISCAL_YEARS.stop + 1)
CORPUS_SPLITS = ["train", "validate", "test"]

# The SEC asks automated clients to identify themselves and to stay under ten
# requests a second.
DEFAULT_USER_AGENT = "academic research script contact@example.com"
SECONDS_BETWEEN_REQUESTS = 0.15


def frame_names() -> list[tuple[str, str]]:
    """Every (tag, period) the link stage needs."""
    names = []
    for year in FRAME_YEARS:
        names += [(ASSETS_TAG, f"CY{year}Q{quarter}I") for quarter in range(1, 5)]
        names += [(tag, f"CY{year}") for tag in [NET_INCOME_TAG, *REVENUE_TAGS]]
    return names


def fetch(settings: Settings) -> None:
    """Download whichever raw files are not yet under `settings.raw_dir`."""
    user_agent = os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)
    targets = [(TICKERS_URL, settings.ticker_list_path)]
    targets += [
        (FRAME_URL.format(tag=tag, period=period), settings.raw_frames_dir / f"{tag}_{period}.json")
        for tag, period in frame_names()
    ]
    missing = [(url, path) for url, path in targets if not path.exists()]
    print(f"{len(targets) - len(missing)} of {len(targets)} SEC files already present")
    for url, path in missing:
        _download(url, path, user_agent)
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    large = [(SUBMISSIONS_URL, settings.submissions_path)]
    large += [
        (
            CORPUS_URL.format(year=year, split=split),
            settings.raw_corpus_dir / f"{year}_{split}.jsonl",
        )
        for year in CORPUS_YEARS
        for split in CORPUS_SPLITS
    ]
    for url, path in large:
        if not path.exists():
            _download_large(url, path, user_agent)


def _download_large(url: str, path: Path, user_agent: str) -> None:
    """Stream `url` to `path` through a `.part` file, so a broken download leaves no file."""
    part = path.with_name(path.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    print(f"  downloading {path.name} ...", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as file:
        shutil.copyfileobj(response, file, length=1 << 20)
    part.rename(path)
    print(f"  downloaded {path.name}")


def _download(url: str, path: Path, user_agent: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        # No filer used this tag in this period. Record that, so the frame is
        # not requested again and the loader sees an empty frame.
        body = json.dumps({"data": []}).encode()
    json.loads(body)  # refuse to store anything that is not JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    print(f"  downloaded {path.name}")


def main() -> None:
    fetch(Settings.from_env())


if __name__ == "__main__":
    main()
