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
| `make all` | Rebuilds every derived table from raw data |
| `make test` | Runs the test suite (needs neither the raw data nor a GPU) |
| `make lint` | Runs the linter, the format check and the type checker |
| `make check` | `lint` and `test` together; what CI runs |

## Data

Raw data is only read, never copied or changed. Its location is one setting,
the `AI_MCDM_RAW_DIR` environment variable (default `~/maids/data`). To try the
pipeline without the raw data, point it at the test fixture:

```
AI_MCDM_RAW_DIR=tests/fixtures/raw make all
```

Derived tables are written under `data/` (ignored by git), each as Parquet
plus a CSV copy to open by hand. `AI_MCDM_DATA_DIR` overrides the location.

| Table | Contents |
|---|---|
| `data/interim/kaggle_clean` | Clean panel: one row per ticker and fiscal year (2014–2018), 221 fundamentals, `sector`, `next_year_return` (%) and `tail_loss` |

`tail_loss` marks the worst 20% of next-year returns within each fiscal year.
Rows whose next-year return is missing, above +1000% or at most -100% are
dropped, because they cannot be labelled.

## Layout

The system is a batch pipeline of stages. Each stage is a function from input
tables to an output table, and stages communicate only through tables.

```
src/ai_mcdm_invest/
  config.py      raw-data and derived-data locations
  pipeline.py    the rebuild command: runs every stage in order
  tables.py      writes a table as Parquet plus CSV
  data/          clean panel, company links, joined panel, filings table
  criteria/      ratio criteria
  ml/            ML risk score
  llm/           LLM extraction from 10-K filings
  mcdm/          weighted sum and TOPSIS
  evaluation/    layer ablation and tests
tests/
  fixtures/      fictional companies with known answers
```
