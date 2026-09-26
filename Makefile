PY := uv run python

MODEL ?= cnn
SUBSET ?= 2000
EPOCHS ?= 30
BATCH ?= 32
LR ?= 0.001
WORKERS ?= 4

.PHONY: setup data labels features train eval pipeline index serve test lint

setup:
	uv sync

data:
	$(PY) -m audiotag.data.download $(if $(SUBSET),--subset $(SUBSET),)
	$(PY) -m audiotag.data.labels

labels:
	$(PY) -m audiotag.data.labels

features:
	$(PY) scripts/precompute_features.py --workers $(WORKERS)

train: features
	$(PY) -m audiotag.train --model $(MODEL) --epochs $(EPOCHS) --batch-size $(BATCH) --lr $(LR) $(if $(SUBSET),--subset $(SUBSET),)

eval:
	$(PY) -m audiotag.eval --model $(MODEL)

pipeline: data features train eval

index:
	$(PY) -m audiotag.index.build --model $(MODEL) --index-type flat
	$(PY) -m audiotag.index.build --model $(MODEL) --index-type ivfpq

benchmark:
	$(PY) scripts/benchmark_index.py

quality:
	$(PY) -m audiotag.index.evaluate

serve:
	docker compose up --build

serve-dev:
	$(PY) -m uvicorn audiotag.api.main:app --reload

loadtest:
	$(PY) scripts/load_test.py $(if $(CONCURRENCY),--concurrency $(CONCURRENCY),) $(if $(DURATION),--duration $(DURATION),)

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
