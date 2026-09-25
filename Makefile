PY := uv run python

.PHONY: setup data labels features train eval index serve test lint

setup:
	uv sync

data:
	$(PY) -m audiotag.data.download $(if $(SUBSET),--subset $(SUBSET),)
	$(PY) -m audiotag.data.labels

labels:
	$(PY) -m audiotag.data.labels

features:
	$(PY) scripts/precompute_features.py $(if $(WORKERS),--workers $(WORKERS),)

train:
	@echo "train: model not implemented yet"

eval:
	@echo "eval: model not implemented yet"

index:
	@echo "index: FAISS index build not implemented yet"

serve:
	@echo "serve: API not implemented yet"

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
