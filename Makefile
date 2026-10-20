.PHONY: setup check smoke doctor

setup:
	UV_CACHE_DIR=$(CURDIR)/.cache/uv uv sync --locked --group dev

check:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	.venv/bin/pytest

smoke:
	.venv/bin/sllm score --gold tests/fixtures/gold.jsonl --predictions tests/fixtures/predictions.jsonl

doctor:
	.venv/bin/sllm doctor
