.PHONY: setup fetch lint test train train-all

setup:
	uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -e ".[dev]"

fetch:               ## download the dataset once (~150 MB) to data/raw/
	.venv/bin/fraud-fetch

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest -q

train:               ## LightGBM candidate
	.venv/bin/fraud-train --config configs/lightgbm.yaml

train-all:
	for c in configs/*.yaml; do .venv/bin/fraud-train --config $$c; done
