"""Download the SEC files the company-link stage reads. Run with `make fetch`.

This is the only code that touches the network. It adds files under
`<raw_dir>/sec/` and never rewrites one that is already there, so the rebuild
(`make all`) stays offline and repeatable.

Files:
- `company_tickers.json`: the SEC's current ticker list.
- `frames/<tag>_<period>.json`: one XBRL "frame" per tag and calendar period,
  holding that figure for every filer that reported it.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.panel import FISCAL_YEARS

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FRAME_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json"

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
    """Download whichever SEC files are not yet under `settings.raw_sec_dir`."""
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
