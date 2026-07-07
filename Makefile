# Local dev entrypoints. `test` pins PYTHONHASHSEED at launch so string-hash
# ordering is deterministic; the root conftest fails fast if it is not pinned.
.PHONY: sync test lint type

# Pin the seed BEFORE the interpreter starts — a fixture cannot do this.
export PYTHONHASHSEED := 0

# The heavy/endpoint Providers are workspace members but NOT in the dev env's
# deps (torch/transformers, apache-airflow, the openai client would break the
# CPU-first, offline test env — ADR-0007/0016). Install ONLY their source, so
# their offline tests (which monkeypatch the heavy dep, or use the Airflow-free
# in-process path) collect and run. Mirrors .github/workflows/ci.yml.
HEAVY_PKGS := \
	latence-ner-gliner latence-relation-gliner latence-relation-llm \
	latence-pii-gliner latence-parser-endpoint latence-ner-endpoint \
	latence-embedder-endpoint latence-runner-airflow

# Every package with a typed source tree (mirrors CI's mypy target list).
ALL_PKGS := \
	latence-core latence-parser-plaintext latence-parser-document \
	$(HEAVY_PKGS) latence-demo

# One-time setup for a fresh checkout: run this before `make test`.
sync:
	uv sync
	$(foreach p,$(HEAVY_PKGS),uv pip install --no-deps -e packages/$(p) &&) true

test:
	uv run pytest

lint:
	uv run ruff check packages

# mypy has no `files` in config; pass the source trees explicitly (mirrors CI).
type:
	uv run mypy --strict $(foreach p,$(ALL_PKGS),packages/$(p)/src)
