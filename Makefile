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
	@echo "index: FAISS index build not implemented yet"

serve:
	@echo "serve: API not implemented yet"

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
