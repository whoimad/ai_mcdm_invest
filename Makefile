.PHONY: all figures notebook fetch manifest test test-slow lint check

# Rebuild every derived table and figure from raw data.
# Raw data is read from AI_MCDM_RAW_DIR (default: ~/maids/data).
all:
	uv run python -m ai_mcdm_invest.pipeline

# Redraw the figures from the tables `make all` wrote, without rebuilding them.
figures:
	uv run python -m ai_mcdm_invest.figures.draw

# Rerun the data notebook from the tables and figures `make all` wrote, saving
# its outputs so it reads on GitHub without the data. Kept out of CI.
notebook:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/data_exploration.ipynb

# Download the raw files the rebuild reads besides the panel: SEC ticker list,
# XBRL frames and bulk submissions into AI_MCDM_RAW_DIR/sec, EDGAR-CORPUS into
# AI_MCDM_RAW_DIR/edgar_corpus (about 16 GB in all). Needs the network; files
# already present are kept.
fetch:
	uv run python -m ai_mcdm_invest.data.sec_download

# Record the raw files the study is built on (size, SHA-256, date) in
# raw_manifest.json. Run only to adopt new raw data on purpose; `make all`
# refuses to run on raw files that differ from the manifest.
manifest:
	uv run python -m ai_mcdm_invest.data.raw_manifest

test:
	uv run pytest

# Acceptance checks on the real raw data (fiscal 2017 join and 10-K coverage). Slow and
# kept out of CI; needs AI_MCDM_RAW_DIR to hold the raw data.
test-slow:
	uv run pytest -m slow

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check

check: lint test
