.PHONY: setup check lint format-check typecheck hooks s3-up s3-down ingest data

DATA_VERSION ?= v1
VALIDATION_SIZE ?= 5000
SEED ?= 42

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

s3-up:
	docker compose up -d rustfs
	docker compose run --rm create-bucket

s3-down:
	docker compose down

ingest:
	uv run --locked ingest-cifar10

data:
	uv run --locked fetch-cifar10-raw
	uv run --locked convert-cifar10
	uv run --locked split-cifar10 --validation-size $(VALIDATION_SIZE) --seed $(SEED)
	uv run --locked stats-cifar10
	uv run --locked publish-cifar10 --version $(DATA_VERSION) --validation-size $(VALIDATION_SIZE) --seed $(SEED)
