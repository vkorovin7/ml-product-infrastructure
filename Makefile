.PHONY: setup check lint format-check typecheck hooks

setup:
	uv sync --locked --group dev
	uv run --locked pre-commit install

check: lint format-check typecheck

lint:
	uv run --locked ruff check .

format-check:
	uv run --locked ruff format --check .

typecheck:
	uv run --locked mypy src scripts

hooks:
	uv run --locked pre-commit run --all-files
