"""SEC submissions: each filer's annual reports and late-filing notices.

The SEC's bulk `submissions.zip` holds one `CIK##########.json` per filer
with its most recent filings, and older filings in further pages
(`CIK##########-submissions-001.json`, ...) named in that file. Each lists
filings as parallel columns: accession number, filing date, report date
(the end of the period the filing covers) and form.
"""

import json
import zipfile
from pathlib import Path

import pandas as pd

# Annual reports (10-K405 until 2003, 10-KT for a transition period) and the
# notice a filer sends when its annual report will be late. Amendments
# (10-K/A) are left out: the filing studied is the original report.
ANNUAL_REPORT_FORMS = {"10-K", "10-K405", "10-KT"}
LATE_NOTICE_FORM = "NT 10-K"
SUBMISSION_COLUMNS = ["cik", "form", "accession", "filing_date", "report_date"]


def load_submissions(path: Path, ciks: set[int]) -> pd.DataFrame:
    """Annual reports and late-filing notices of the filers in `ciks`.

    Columns: `cik`, `form`, `accession`, `filing_date` and `report_date`
    (dates; the report date is empty when the SEC lists none). Filers missing
    from the archive have no rows.
    """
    if not path.is_file():
        raise FileNotFoundError(f"SEC submissions not found: {path}. Run `make fetch` first.")
    wanted = ANNUAL_REPORT_FORMS | {LATE_NOTICE_FORM}
    frames = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for cik in sorted(ciks):
            main = f"CIK{cik:010d}.json"
            if main not in names:
                continue
            body = json.loads(archive.read(main))
            pages = [body["filings"]["recent"]]
            pages += [
                json.loads(archive.read(page["name"]))
                for page in body["filings"].get("files", [])
                if page["name"] in names
            ]
            for page in pages:
                frame = pd.DataFrame(
                    {
                        "form": page["form"],
                        "accession": page["accessionNumber"],
                        "filing_date": page["filingDate"],
                        "report_date": page["reportDate"],
                    }
                )
                frames.append(frame[frame["form"].isin(wanted)].assign(cik=cik))
    if not frames:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS).astype(
            {"cik": "int64", "filing_date": "datetime64[ns]", "report_date": "datetime64[ns]"}
        )
    filings = pd.concat(frames, ignore_index=True)
    for column in ["filing_date", "report_date"]:
        filings[column] = pd.to_datetime(filings[column].replace("", None))
    return filings[SUBMISSION_COLUMNS].drop_duplicates("accession", ignore_index=True)
