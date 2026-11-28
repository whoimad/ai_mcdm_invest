# ai_mcdm_invest

Does adding intelligence to multi-criteria decision making (MCDM) improve a
risk-aware stock ranking? The study compares three layers of criteria (six
textbook ratios, an ML risk score, and facts an LLM extracts from 10-K
filings) on fiscal years the text pipeline never saw.

**Start here:** [`notebooks/data_exploration.ipynb`](notebooks/data_exploration.ipynb)
is a tour of the data and what it showed before any model was built. It is saved
with its results, so it reads on GitHub without the data.

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
| `make results` | After `make all` and the test-year going-concern run: the "+ LLM" ablation row, the three likelihood-ratio tests and the tables beside them, into `data/processed/test_year_*` and `data/reports/test_year_results.md` |
| `make screener` | Opens the screener in the browser: the study's ranking on the two test years, with a company card (see below) |
| `make figures` | Redraws the figures from the tables `make all` wrote, without rebuilding them |
| `make notebook` | Reruns the data notebook from those tables and figures and saves its outputs; kept out of CI |
| `make test` | Runs the test suite (needs neither the raw data nor a GPU) |
| `make test-slow` | Runs the acceptance checks on the real raw data (fiscal 2017 join, 10-K coverage, the ML model's AUC); kept out of CI |
| `make lint` | Runs the linter, the format check and the type checker |
| `make check` | `lint` and `test` together; what CI runs |

## Screener

`make screener` (or `uv run streamlit run app/screener.py`) opens a
read-only view of the study's ranking in the browser. It reads the tables in
`data/processed` and calls no model. Its logic lives in
`ai_mcdm_invest.screener`; the scores come from the package's weights, SAW
and TOPSIS, and the theme from `.streamlit/config.toml` (one colour for risk,
one for safety).

The top bar holds:

- **As of**: "Fiscal 2017, returns over 2018" or "Fiscal 2018, returns over
  2019". Companies are ranked on their fiscal-year data.
- **Method**: SAW (the study's main ranker, the weighted sum of the
  percentiles, 0 to 100) or TOPSIS (its robustness check, the closeness to the
  ideal company, 0 to 1). TOPSIS takes its scale, ideal and worst from the
  study's evaluation universe, both test years together, as the study does.
- **Layers**: ratios, ML risk model, 10-K text. A layer switched off gives its
  groups no weight, greys out their sliders, and the rest are rescaled. The
  text layer is left out until the test-year going-concern run has written its
  tables.
- **Reveal next-year returns** (off by default): the return column and the
  share of tail losers in the riskiest and safest fifths of the rows shown.
- **Profile**: Balanced, Risk-averse, Value-minded (as in the table above) or
  Custom, with one slider per group (Profitability, Valuation, Balance sheet,
  ML risk, 10-K text). A profile fills the sliders; moving one switches to
  Custom; "Reset" goes back to the last profile. Weights are rescaled to 100%
  and split equally within a group. On a fixed profile with the study's layer
  rows (ratios only, + ML, + LLM) the page says the scores match the study's;
  on Custom it says the weights are not part of the study's tests.

Filters (sector, going-concern status, ticker or name) only hide rows: scores
and ranks stay those of the whole year. Clicking a row opens its card: rank and
score, then tabs for the ratios (scored market-wide percentile, the sector
percentile as a grey mark, context, not scored, and each criterion's current
weight), the ML risk (safety percentile, tail-loss probability, the five
largest contributions from `ml_risk` as signed bars) and the 10-K text (the
auditor's quoted paragraph when flagged, "no flags" when clear, and "text not
available" with the reason otherwise). Only scored facts are shown.

The screener lists every test-year company with enough ratios; the study's
measures use only those whose 10-K text is readable. `make test-slow` checks
that on every fixed profile, layer row and method its scores give the study's
tail-loss AUC on that universe, and runs the page through `streamlit.testing`.
It ranks scored facts under fixed weights: it does not recommend buying any
stock and does not forecast returns.

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
AI_MCDM_RAW_DIR=tests/fixtures/raw AI_MCDM_MANIFEST=tests/fixtures/raw_manifest.json \
  AI_MCDM_WORKED_EXAMPLE=ROMO:2015 make all
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
| `data/processed/criterion_directions` | Each ratio criterion's `group`, textbook direction, `training_auc` and `flipped_training_auc`, and learned `direction` (`keep`, `flip`, `drop`) |
| `data/processed/ratio_criteria` | Ratio criteria: one row per joined-panel row, each criterion's `<name>_score`, `<name>_sector_score` and `<name>_status`, with `ratios_not_reported` and `too_few_ratios` (see below) |
| `data/processed/ml_risk` | ML risk score: one row per clean-panel row in the test years (`ticker`, `fiscal_year`), with `tail_loss_probability` and the five largest feature contributions, `feature_1`–`feature_5` and `contribution_1`–`contribution_5` (see below) |
| `data/processed/ml_criterion` | The ML risk criterion: one row per joined-panel row in the test years, with `tail_loss_probability` and `ml_risk_score` |
| `data/processed/baselines` | Size (`market_cap`) and last year's price move (`last_year_return`, %) for each joined-panel row, kept outside MCDM |
| `data/processed/ablation` | The layer ablation: one row per `layer`, `profile`, `method` and `fiscal_year` (2017, 2018, `pooled`), plus the baselines, with `exploratory`, `companies`, `tail_losers` and the four measures (see below) |
| `data/processed/worked_example` | The 10-K passages the worked-example figure quotes: `ticker`, `cik`, `fiscal_year`, `item`, `passage` |
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
"going concern", `;`-separated), `fallback_read` (no auditor's report
located, see below), `no_report` (no audit opinion text in Items 8 and 15
at all), `item1a_changed_share`, then
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
EDGAR-CORPUS also lacks some very large companies (AT&T, GE, Pfizer,
Verizon); the "10-K in the corpus" step excludes and counts them.

An Item counts as absent when its section is empty or a stub: under 1,000
characters and opening by saying it is omitted, not applicable, or found
elsewhere ("see Item 15", "beginning on page F-1"). Many Item 8 sections are
such stubs, the statements being placed in Item 15. `item1a_changed_share`
is the share of this year's Item 1A paragraphs of which fewer than half the
sentences appear word for word in the prior year's Item 1A; it is empty when
either year has no Item 1A.

### 10-K text

A 10-K is the annual report a US public company files with the SEC. Its
layout is set by the SEC: numbered Items, the same for every company. The
ones this study uses:

| Item | Contents | Used for |
|---|---|---|
| 1 | Business | Presence, length |
| 1A | Risk factors | `item1a_changed_share` |
| 7 | Management's discussion and analysis (MD&A) | Period end, length |
| 8 | Financial statements, usually with the auditor's report | Period end, auditor's report |
| 15 | Exhibits and schedules; some companies put the statements and the auditor's report here instead of in Item 8 | Period end, auditor's report |

On EDGAR the official filing is an HTML page (plain text with SGML markup
before about 2001; Inline XBRL, still HTML, since 2019). A PDF copy may be
attached but is not the official filing. The study never reads the HTML.
EDGAR-CORPUS has already turned each 10-K into plain text, split by Item,
one JSON line per filing:

```
{"filename": "1514994_2016.htm", "cik": "1514994", "year": "2016",
 "section_1": "...", "section_1A": "...", ..., "section_8": "...", "section_15": "..."}
```

The conversion loses some things, and the code is written around that:

- **Tables are mostly gone.** A statement keeps its title ("CONSOLIDATED
  STATEMENTS OF CASH FLOWS") but not its numbers, so financial figures come
  from the Kaggle panel and the XBRL frames, never from the text.
- **Lines can be dropped or merged.** Some auditor's reports lost their
  signature and date, so a missing date does not drop a report.
- **The `year` label is not reliable**, so the period is read from the text
  (see above).

From a corpus record to what the LLM reads:

1. `data/corpus.py` turns the `section_<Item>` fields into an Item → text
   map, flags stub Items as absent, reads the period end, and keeps only
   facts (presence, length, going-concern wording). Raw text stays in the
   corpus files.
2. `llm/locate.py` finds the auditor's report in Items 8 and 15 by
   document structure: the heading "Report of Independent Registered
   Public Accounting Firm" down to the auditor's signature date. A heading
   counts only when the opinion opens within 1,500 characters after it ("We
   have audited", or "In our opinion"), so index entries and
   cross-references are skipped. The report ends at the first line that
   starts with a date, else at the next heading.
3. Each report is its own chunk of at most about 6k tokens (24,000
   characters). A 10-K can have several: a separate report on internal
   control, or one from a former auditor.
4. When no report is found (asset-backed trusts, statements incorporated by
   reference), both Items are read in full in 6k-token chunks cut between
   lines, and the 10-K is marked `fallback_read`, so its results can be
   shown apart. When Items 8 and 15 hold no audit opinion text at all ("we
   have audited", or "in our opinion" followed by "present fairly" in the same
   sentence), the 10-K is also marked `no_report`: nothing is sent to the LLM,
   and it is left out of the evaluation universe (decision 5.2). A bare "in our
   opinion" is not enough, since management writes it too ("In our opinion, we
   have made adequate provisions for income taxes"). Of the 8,926 fiscal
   2014–2016 10-Ks in the corpus, 266 (3.0%) are `no_report`, and 118 (1.3%)
   are fallback reads that hold opinion text and are read in full.
5. `llm/going_concern.py` sends each chunk to the LLM, which returns the
   going-concern passage word for word, then five yes/no facts about it: is it
   the auditor's opinion, about this company, about the fiscal year the 10-K
   covers, does it express doubt, did management's plans alleviate it. Code
   checks the quote is in the chunk (ignoring whitespace). An answer that fails
   a check is asked again twice, then recorded as `extraction_failed`, never as
   clear. A chunk is `flagged` only when the first four facts are all yes; the
   fifth is kept for error analysis. A 10-K is flagged when any chunk is, else
   failed when any chunk failed, else clear. Every reply is cached by prompt
   version, model version and prompt text, and each finished 10-K leaves a
   marker, so an interrupted run resumes where it stopped.

Training-year run (`python -m ai_mcdm_invest.llm.training_run`): every
fiscal 2014–2016 10-K that meets the evaluation universe's rule (in the
corpus, at least half the ratios), read by `qwen2.5:7b` (digest
`845dbda0ea48`):

| Fiscal year | 10-Ks | Flagged | Clear | Failed | No report | Fallback reads | Seconds per 10-K |
|---|---|---|---|---|---|---|---|
| 2014 | 2,750 | 64 | 2,599 | 2 | 85 | 36 | 2.55 |
| 2015 | 2,989 | 101 | 2,794 | 3 | 91 | 43 | 2.64 |
| 2016 | 3,133 | 135 | 2,905 | 4 | 89 | 39 | 2.62 |
| All | 8,872 | 300 | 8,298 | 9 | 265 | 118 | 2.61 |

The 8,872 10-Ks are the 8,926 in the corpus less those with too few ratios;
one `no_report` 10-K is among the latter. The 118 fallback reads are counted
inside flagged, clear and failed. Checked by hand, all four flagged fallback
reads hold a genuine auditor's going-concern paragraph.

Test-year run (`uv run python -m ai_mcdm_invest.llm.test_year_run`): the same
code, universe rule, locate rule, model and prompt version on fiscal 2017 and
2018, with `kw-1` on the same located blocks. It refuses to start unless
`PREREGISTRATION.md` is committed with no uncommitted changes and its last
commit is on a remote branch; that commit is stored in every row
(`prereg_commit`) and in `data/reports/going_concern_test_run.md`, beside the
counts and the list of "extraction failed" 10-Ks. Its store
(`data/interim/going_concern_test_store`) is apart from the training run's, and
a killed run resumes only under the same prompt version and model digest.
Results: `going_concern_test_filings` and `going_concern_test_chunks`.

Run on 2026-11-27/28 with prompt `gc-2`, after `gc-2` passed the gate (200 of
200 canaries) and `PREREGISTRATION.md` was pushed (commit `4947c87`). The
training-year run above and `kw-1`'s tuning used `gc-1`; they were not rerun.

| Fiscal year | 10-Ks | Flagged | Clear | Failed | No report | Fallback reads | Seconds per 10-K |
|---|---|---|---|---|---|---|---|
| 2017 | 3,343 | 145 | 3,098 | 16 | 84 | 35 | 3.25 |
| 2018 | 3,357 | 168 | 3,080 | 23 | 86 | 36 | 3.52 |
| All | 6,700 | 313 | 6,178 | 39 | 170 | 71 | 3.39 |

Seconds per 10-K are higher than on the training years partly because the run
was paused whenever the GPU reached 85 °C.

The keyword benchmark (`llm/keyword_benchmark.py`, decision 6.6) is the
non-LLM rival. A strict regex reads the same located chunks one sentence at a
time. It flags a sentence that says there is substantial doubt about a named
company's ability to continue as a going concern, unless the sentence uses the
accounting rule's own wording ("whether", "would", "evaluate", "an entity's").
It was tuned on fiscal 2014–2016 against the LLM's flags and frozen as `kw-1`.
Its flags are stored beside the LLM's (`keyword_flagged`, and the matched
sentence as `keyword_match`). Of the 8,598 flagged or clear 10-Ks, it flags
316: all 300 the LLM flags, and 16 it does not. It also flags 2 of the 9
failed 10-Ks. By design it cannot see two traps: a reprinted opinion from an
earlier year, and another company's doubt.

A located report, shortened (North Texas Energy, fiscal 2016):

```
REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM      <- heading
To the Board of Directors and Stockholders of North Texas Energy, Inc.
We have audited the accompanying consolidated balance sheets  <- opening
...
In our opinion, the financial statements referred to above present fairly ...
... These factors raise substantial doubt about its ability to continue as
a going concern. ...
/s/ GBH CPAs, PC
Houston, Texas
September 29, 2017                                            <- end
```

## Ratio criteria

The classic MCDM layer is six textbook ratios in three groups: profitability
(`roe`, return on equity; `net_margin`), valuation (`pe`, price-to-earnings)
and balance sheet (`debt_equity`, `current_ratio`, `interest_coverage`). Each
is scored as its percentile within the fiscal year over the whole joined
panel (`<name>_score`), oriented so that higher is better. Its percentile
within its sector and year (`<name>_sector_score`) is kept beside it as
context; it is not scored.

`<name>_status` says how each ratio was read:

| Status | Meaning | Score |
|---|---|---|
| `reported` | A number, ranked with the others | its percentile |
| `negative earnings` | Price-to-earnings with net income below zero, or a ratio at or below zero (the source writes 0 for a loss) | worst |
| `negative equity` | Return on equity or debt-to-equity with shareholders' equity below zero | worst |
| `no interest expense` | Interest coverage of 0 for a company paying no interest (the coverage is really unbounded) | best |
| `not reported` | Missing | 0.5 |

Odd values stay worst or best whichever way the criterion points, and several
worst rows share the lowest rank. A ratio missing for a bad reason (say, no
return on equity because equity is negative) is scored worst, not 0.5. A company
with more than three of the six ratios not reported is marked
`too_few_ratios` and leaves the evaluation; the rebuild prints how many per
year, with their tail losers.

Each criterion's direction is learned on the training years (fiscal
2014–2016) from its tail-loss AUC: the chance that a company avoiding a tail
loss scores higher than a tail loser (ties count half; rows with the ratio not
reported are left out). The AUC is taken twice, with the textbook direction
(`training_auc`) and flipped (`flipped_training_auc`), odd values staying worst
or best both ways. The criterion is used the way with the higher AUC if that
AUC is at least 0.55, and dropped (no score) otherwise. Leverage is why both
ways are tried: negative equity is the riskiest, but among the other companies
more debt goes with fewer tail losses (the `leverage_tail_loss` figure).
Taken one way only, the two cancel out and debt-to-equity would be dropped.

## ML risk score

The ML layer is one criterion: the probability that a company lands in its
year's worst 20% of next-year returns (a tail loss). It comes from
gradient-boosted trees (XGBoost) on the 221 fundamentals, with the
prototype's settings, never tuned: 400 trees, depth 4, learning rate 0.03,
row subsample 0.8, column subsample 0.5, minimum child weight 10, histogram
splits, seed 0. Unlike the prototype, the fundamentals go in as they are: no
added features, no columns dropped for missing values, no capping of extreme
values; missing values are left to the trees.

The model is trained walk-forward on the full clean panel, linked to a 10-K
or not: on fiscal 2015–2016 to score 2017, and on 2015–2017 to score 2018. Its
out-of-time tail-loss AUC is 0.810 (2017) and 0.792 (2018); the prototype,
which also fed it sector and last year's price move, scored 0.813 and 0.796.
`make test-slow` checks the rewrite stays within 0.01 of 0.81 and 0.80.

First result of adding the ML layer (exploratory; pooled 2017–2018, SAW,
6,530 companies):

| Profile | Tail-loss AUC, ratios only | Tail-loss AUC, + ML | Tail losers in the safest fifth, ratios only → + ML |
|---|---|---|---|
| Balanced | 0.705 | 0.748 | 10.3% → 5.0% |
| Risk-averse | 0.706 | 0.775 | 8.2% → 2.3% |
| Value-minded | 0.699 | 0.738 | 11.9% → 7.1% |

The baselines score 0.707 (size) and 0.622 (last year's price move).

`ml_risk_score` is the probability's percentile within the fiscal year over
the joined panel, as the ratio scores are, oriented so that higher is safer.
Each company keeps the five features whose contributions to its score are
largest in size, from XGBoost's own contribution output (SHAP values, in
log-odds of a tail loss; positive raises the risk).

**The ML results are exploratory.** The settings were never tuned on the test
years, but the target and the model family were chosen while looking at them.
The ablation's `exploratory` column marks every row that includes the ML layer.

## Ranking and the ablation

The ranker is the weighted sum (SAW): each company's weighted mean of its
criteria. TOPSIS, the closeness to an ideal company, is run beside it as a
robustness check. Weights come from three profiles fixed before testing, by
criterion group, and are never fitted to returns:

| Profile | Profitability | Valuation | Balance sheet | ML risk | Text |
|---|---|---|---|---|---|
| Balanced | 20% | 20% | 20% | 20% | 20% |
| Risk-averse | 10% | 10% | 20% | 40% | 20% |
| Value-minded | 25% | 35% | 10% | 20% | 10% |

A group's weight is split equally over its criteria in use (a dropped ratio
leaves its share to the others in its group). A group with no criterion in use
(its layer switched off, or not built yet) loses its weight, and the rest are
rescaled to 100%. A company without a value for a criterion is ranked on its
remaining criteria. A constant criterion does not break either ranker: it adds
the same to everyone in SAW and nothing in TOPSIS.

The ablation measures every row on the same companies, the evaluation
universe: fiscal 2017 and 2018 rows whose 10-K text is in the corpus and
whose Items 8 and 15 hold audit opinion text, less those with too few ratios.
Size (bigger is safer) and last year's price move (a rise
is safer) are ranked beside the layers as baselines that cost nothing; a
company with no prior-year row, or no market cap, is put in the middle of its
year. Each ranking is described per test year and pooled, with scores and
returns ranked within each year first:

| Column | Meaning |
|---|---|
| `tail_loss_auc` | The chance that a company avoiding a tail loss ranks safer than a tail loser (ties count half) |
| `riskiest_fifth_tail_share`, `safest_fifth_tail_share` | The share of the tail losers that fall in each year's riskiest or safest fifth (20% each at random) |
| `rank_correlation` | Rank correlation of the score with the next-year return |
| `top_minus_bottom_return` | Mean next-year return of the safest fifth minus the riskiest, in percentage points |

`make all` writes the `ratios only` and `+ ML` rows. `make results` adds the
`+ LLM` row once the test-year going-concern run has written its tables: the
text group's one criterion is the LLM's flag (clear scores 1, flagged 0; a
failed extraction has no score and is ranked on its other criteria). It is
written with every row twice (`sample`): on the whole universe and without
fallback reads, SAW and TOPSIS both.

The primary tests (pre-registration, section 4) are likelihood-ratio tests,
pooled 2017–2018, at α = 0.05, on the companies whose extraction did not fail.
Each adds one flag to a logistic model of tail loss on the logit of the ML
tail-loss probability and a 2018 dummy (each year's ML score comes from its own
walk-forward model, so each year keeps its own intercept): H1 the LLM flag, H2
the `kw-1` flag, H3 the LLM flag given the `kw-1` flag. Each gives the LR
statistic (1 degree of freedom), its p-value and the flag's odds ratio with a
95% Wald interval (`test_year_tests`). Descriptive only, without intervals: the
AUC gain of `+ LLM` over `+ ML` (`test_year_auc_gain`), the flagged companies
already in ML's riskiest fifth (`test_year_riskiest_fifth`), and the filings
where the LLM and `kw-1` disagree, with both quotes (`test_year_disagreements`,
the ten ML finds riskiest shown in the summary). Failed extractions are listed
in `test_year_failed`.

Results on fiscal 2017–2018 pooled (6,530 companies, 1,410 tail losers; the
tests use the 6,491 whose extraction did not fail):

| Test | Flag | Given | Flagged | LR statistic | p-value | Odds ratio (95% interval) |
|---|---|---|---|---|---|---|
| H1 | LLM | ML | 313 | 30.5 | < 0.001 | 2.13 (1.62–2.80) |
| H2 | `kw-1` | ML | 311 | 27.2 | < 0.001 | 2.04 (1.55–2.68) |
| H3 | LLM | ML, `kw-1` | 313 | 3.3 | 0.071 | 2.03 (0.93–4.42) |

Both flags add information beyond the ML score, but the LLM adds no
significant information beyond the keyword benchmark (H3 is a null). 88% of
the LLM's flagged companies are already in ML's riskiest fifth (276 of 313),
and their tail-loss rate is 68%. The AUC gain of `+ LLM` over `+ ML` is 0.004
to 0.007 across profiles, SAW and TOPSIS, with or without fallback reads
(balanced SAW: 0.748 to 0.754). The LLM and `kw-1` disagree on 30 filings (16
flagged by the LLM only, 14 by `kw-1` only). Among the LLM-only flags are
errors: a management note quoting the accounting rule (ASU 2014-15), and
auditor paragraphs that only say the company "will require additional
financing", with no substantial doubt stated. The 39 failed extractions have a
tail-loss rate of 59%.

## Figures

`make all` ends by drawing the study's figures into `data/figures/`, each as
PDF and PNG, with every caption (sample size, source, fiscal
and return years) in `data/figures/captions.md`. They read only the pipeline's
tables, and describe the data only: no model result appears in them. The four
storyline figures carry the study's argument:

| Figure | Shows |
|---|---|
| `join_funnel` | Companies and tail losers left at each step, from the clean panel through the ticker and figure links, one row per filer, the 10-K and its text, to Items 1A, 7 and 8 or 15 present and the auditor's opinion text in them |
| `leverage_tail_loss` | Tail-loss rate by debt-to-equity quintile, cut across the market and within sectors, with negative equity apart; beside it each sector's median debt-to-equity against its tail-loss rate |
| `section_lengths` | Estimated tokens of Items 1A, 7 and 8 (stubs left out) against the 32,768-token context window, with the share over it |
| `going_concern_items` | How many 10-Ks have "going concern" in each Item, Items 8 and 15 (statements and auditor's report) set apart |

The rest of the exploratory section:

| Figure | Shows |
|---|---|
| `companies_per_sector` | Companies (SEC filers) per sector, with their company-years |
| `missing_values` | Share of fundamentals missing, by column group (the source's six statements) and sector |
| `market_cap` | Market-cap histogram on a log scale, with the median and the usual size bounds |
| `valuation_by_sector` | Price-to-earnings and price-to-book by sector (box plots, log scale), with the share of losses |
| `criteria_correlation` | Rank correlation of the six textbook criteria, ranked within each fiscal year |
| `return_distribution` | Next-year returns per fiscal year, with each year's tail-loss cut-off |
| `tail_loss_by_size_sector` | Tail-loss rate by market-cap quintile and by sector |
| `item1a_length` | Item 1A length per fiscal year, for all filers and for those present every year |
| `filing_lag` | Days from fiscal year end to the 10-K, tail losers against others, with the SEC deadlines |
| `item1a_change` | Share of Item 1A paragraphs new or changed since the prior year |
| `worked_example` | One company's six criteria and their percentile beside quoted 10-K passages |

The worked example is chosen by hand from the years before the test years:
Caesars Entertainment, fiscal 2014, whose auditor doubted it could continue as
a going concern. It is a setting, `AI_MCDM_WORKED_EXAMPLE` (`TICKER:YEAR`,
default `CZR:2014`), because it must be in the raw data the build reads; the
fixture dataset uses `ROMO:2015`. Its passages are read from EDGAR-CORPUS during `make all` and kept in
`data/processed/worked_example`, so `make figures` needs no raw data.

## Layout

The system is a batch pipeline of stages. Each stage is a function from input
tables to an output table, and stages communicate only through tables.

```
src/ai_mcdm_invest/
  config.py      raw-data and derived-data locations
  pipeline.py    the rebuild command: runs every stage in order
  tables.py      writes a table as Parquet plus CSV
  data/          clean panel, company links, joined panel, filings table,
                 worked-example passages
  figures/       report figures drawn from the tables
  criteria/      ratio criteria and their learned directions
  ml/            ML risk score
  llm/           LLM extraction from 10-K filings
  mcdm/          weighted sum, TOPSIS and the weight profiles
  evaluation/    baselines, evaluation universe, layer ablation and test-year results
tests/
  fixtures/      fictional companies with known answers
notebooks/
  data_exploration.ipynb   a tour of the data, saved with its results
```
