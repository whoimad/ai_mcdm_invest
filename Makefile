.PHONY: all fetch test lint check

# Rebuild every derived table from raw data.
# Raw data is read from AI_MCDM_RAW_DIR (default: ~/maids/data).
all:
	uv run python -m ai_mcdm_invest.pipeline

# Download the SEC files the rebuild reads (ticker list, XBRL frames) into
# AI_MCDM_RAW_DIR/sec. Needs the network; files already present are kept.
fetch:
	uv run python -m ai_mcdm_invest.data.sec_download

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check

check: lint test
