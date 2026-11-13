.PHONY: all fetch manifest test test-slow lint check

# Rebuild every derived table from raw data.
# Raw data is read from AI_MCDM_RAW_DIR (default: ~/maids/data).
all:
	uv run python -m ai_mcdm_invest.pipeline

# Download the SEC files the rebuild reads (ticker list, XBRL frames) into
# AI_MCDM_RAW_DIR/sec. Needs the network; files already present are kept.
fetch:
	uv run python -m ai_mcdm_invest.data.sec_download

# Record the raw files the study is built on (size, SHA-256, date) in
# raw_manifest.json. Run only to adopt new raw data on purpose; `make all`
# refuses to run on raw files that differ from the manifest.
manifest:
	uv run python -m ai_mcdm_invest.data.raw_manifest

test:
	uv run pytest

# Acceptance checks on the real raw data (fiscal 2017 join coverage). Slow and
# kept out of CI; needs AI_MCDM_RAW_DIR to hold the raw data.
test-slow:
	uv run pytest -m slow

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check

check: lint test
