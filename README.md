# ai_mcdm_invest

Does adding intelligence to multi-criteria decision making (MCDM) improve a
risk-aware stock ranking? The study compares three layers of criteria (six
textbook ratios, an ML risk score, and facts an LLM extracts from 10-K
filings) on fiscal years the text pipeline never saw.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
```

## Commands

| Command | What it does |
|---|---|
| `make fetch` | Downloads the SEC files and the 10-K corpus the rebuild reads, about 16 GB (needs the network; run once) |
| `make manifest` | Records the raw files the study is built on in `raw_manifest.json` (only to adopt new raw data on purpose) |
| `make all` | Checks the raw files against `raw_manifest.json`, then rebuilds every derived table and figure, offline |
| `make figures` | Redraws the figures from the tables `make all` wrote, without rebuilding them |
| `make test` | Runs the test suite (needs neither the raw data nor a GPU) |
| `make test-slow` | Runs the acceptance checks on the real raw data (fiscal 2017 join and 10-K coverage); kept out of CI |
| `make lint` | Runs the linter, the format check and the type checker |
| `make check` | `lint` and `test` together; what CI runs |

## Data

Raw data is only read, never copied or changed. Its location is one setting,
the `AI_MCDM_RAW_DIR` environment variable (default `~/maids/data`). It holds:

| Folder | Contents | Source |
|---|---|---|
| `2014 2018/` | Kaggle "200+ Financial Indicators of US Stocks", one CSV per fiscal year | Downloaded by hand |
| `sec/company_tickers.json` | Today's SEC ticker list | `make fetch` |
| `sec/frames/` | XBRL frames for calendar 2013-2019: total assets, revenue (7 tags), net income, for every filer | `make fetch` |
| `sec/submissions.zip` | Every filer's filing history: form, filing date, report date | `make fetch` |
| `edgar_corpus/` | EDGAR-CORPUS (Hugging Face `eloukas/edgar-corpus`): 10-Ks split by Item, one JSON line each, files labelled 2012-2019 | `make fetch` |

`make fetch` only adds files that are missing and never rewrites one. The SEC
asks automated clients to name themselves; set `SEC_USER_AGENT` to override the
generic default.

**The study is tied to one copy of the raw data.** SEC data about 2014-2018
still changes: companies restate figures (the XBRL frames keep the latest
filed value), old filings carry today's company names, and the ticker list
changes daily. So `raw_manifest.json` (committed) records the size, SHA-256
and date of every file in the raw folders the build reads; the ticker list
and XBRL frames are the copy downloaded on 2026-11-12, the submissions and
the corpus the copy downloaded on 2026-11-13. `make all` stops before writing anything if
a file is missing, changed or unlisted. To build on new raw data, run
`make manifest` and commit the change, so the switch is visible in git. To try the
pipeline without the raw data, point it at the test fixture and its manifest:

```
AI_MCDM_RAW_DIR=tests/fixtures/raw AI_MCDM_MANIFEST=tests/fixtures/raw_manifest.json make all
```

Derived tables are written under `data/` (ignored by git), each as Parquet
plus a CSV copy to open by hand, and reports as Markdown under `data/reports/`.
`AI_MCDM_DATA_DIR` overrides the location.

| Table | Contents |
|---|---|
| `data/interim/kaggle_clean` | Clean panel: one row per ticker and fiscal year (2014–2018), 221 fundamentals, `sector`, `next_year_return` (%) and `tail_loss` |
| `data/interim/company_links` | Each clean-panel row's SEC filer: `cik`, `link_method` (`figure_match`, `ticker`, `unlinked`), `figure_match` (`matched`, `ambiguous`, `none`), the CIK each method found, and the match evidence (filer name, period end, which figures matched) |
| `data/processed/panel` | Joined panel: one row per SEC filer (`cik`) and fiscal year, with the kept row's `ticker` and `link_method`, `sector`, the 221 fundamentals, `next_year_return` and `tail_loss` |
| `data/processed/filings` | Filings table: one row per joined-panel row, describing the 10-K for its period (see below) |
| `data/reports/join_report.md` | Companies and tail losers kept and lost at each join step per year, the fiscal 2017 coverage, and 20 random figure-match recoveries to check by eye |

`tail_loss` marks the worst 20% of next-year returns within each fiscal year.
Rows whose next-year return is missing, above +1000% or at most -100% are
dropped, because they cannot be labelled.

`company_links` has one row per clean-panel row. A row is linked by its own
figures first: total assets, plus revenue or net income, must equal one SEC
filer's XBRL figures within 0.1%, for a period ending in the same fiscal year.
Today's ticker list is used only when the figures find no filer; where both
find one and they differ, the figures win (the ticker was usually reused by a
later company). A match to several filers is rejected as `ambiguous`; the
row then keeps its ticker link only if that filer is one of those matched.

The joined panel leaves out unlinked rows and keeps one row per filer and
year. Several rows share a filer when the panel lists a company's other share
classes, preferred shares or notes with the company's own figures, or when a
reused ticker points at a filer another row's figures matched. The kept row is
the first of: rows with a market cap (preferred shares and notes have none),
rows linked by figures, the largest market cap, the shortest ticker. The
panel records no security type, so a preferred share or note that is its
filer's only row in a year cannot be told apart and stays in (CHS Inc., listed
only through its preferred shares, is one).

The build stops before writing the joined panel if fiscal 2017 links fewer
than 85% of companies or 88% of tail losers (the prototype linked 87.6% and
90.6%); the join report is written first, to show where the join fell short.

The filings table describes each joined row's 10-K: `period_end`,
`accession`, `filing_date`, `days_after_fy_end`, `days_after_jan1` (days from
1 January of the return year, `fiscal_year + 1`; negative when the 10-K was
public before the return year began), `december_year_end`, `nt_10k` (a
late-filing notice was filed), `in_corpus` and `corpus_year` (where its text
is), `has_item_{1,1A,3,7,8,9A,15}`, `tokens_item_{1,1A,7,8,15}` and
`tokens_total` (characters ÷ 4), `gc_keyword_items` (Items containing
"going concern", `;`-separated), `item1a_changed_share`, then
`next_year_return` and `tail_loss`.

The 10-K is chosen by the period it reports on, never by filing order: a row
linked by figures takes the 10-K whose SEC report date is within 7 days of
its matched XBRL period end, a row linked by ticker the 10-K whose report
date falls in its fiscal year. The corpus labels are not reliable (Amazon's
fiscal 2018 10-K is labelled 2019, Walmart's year ending January 2018 is
labelled 2018), so each corpus record's period is read from its own text, as
the date Items 7, 8 and 15 mention most, and matched to that report date.
Rows without such a 10-K (foreign filers, funds, companies whose first 10-K
came later) or whose 10-K is not in the corpus keep their row with those
columns empty; fiscal 2017 has a 10-K for 81.5% of rows and its text for
79.5%.

An Item counts as absent when its section is empty or a stub: under 1,000
characters and opening by saying it is omitted, not applicable, or found
elsewhere ("see Item 15", "beginning on page F-1"). Many Item 8 sections are
such stubs, the statements being placed in Item 15. `item1a_changed_share`
is the share of this year's Item 1A paragraphs of which fewer than half the
sentences appear word for word in the prior year's Item 1A; it is empty when
either year has no Item 1A.

## Figures

`make all` ends by drawing the report's four storyline figures into
`data/figures/`, each as PDF (for the report) and PNG, with every caption
(sample size, source, fiscal and return years) in `data/figures/captions.md`.
They read only the pipeline's tables:

| Figure | Shows |
|---|---|
| `join_funnel` | Companies and tail losers left at each step, from the clean panel through the ticker and figure links, one row per filer, the 10-K and its text, to Items 1A, 7 and 8 or 15 present |
| `leverage_tail_loss` | Tail-loss rate by debt-to-equity quintile, cut across the market and within sectors, with negative equity apart; beside it each sector's median debt-to-equity against its tail-loss rate |
| `section_lengths` | Estimated tokens of Items 1A, 7 and 8 (stubs left out) against the 32,768-token context window, with the share over it |
| `going_concern_items` | How many 10-Ks have "going concern" in each Item, Items 8 and 15 (statements and auditor's report) set apart |

## Layout

The system is a batch pipeline of stages. Each stage is a function from input
tables to an output table, and stages communicate only through tables.

```
src/ai_mcdm_invest/
  config.py      raw-data and derived-data locations
  pipeline.py    the rebuild command: runs every stage in order
  tables.py      writes a table as Parquet plus CSV
  data/          clean panel, company links, joined panel, filings table
  figures/       report figures drawn from the tables
  criteria/      ratio criteria
  ml/            ML risk score
  llm/           LLM extraction from 10-K filings
  mcdm/          weighted sum and TOPSIS
  evaluation/    layer ablation and tests
tests/
  fixtures/      fictional companies with known answers
```
