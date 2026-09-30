.PHONY: setup up down migrate test lint typecheck eval bench demo

setup:  ## Install dependencies
	uv sync

up:  ## Start Postgres (pgvector) and Redis
	docker compose up -d

down:  ## Stop the services (data volume is kept)
	docker compose down

migrate:  ## Apply database migrations (as the owner role)
	uv run python -m app.db.migrate

test:  ## Unit and integration tests (no live API calls, no cost)
	uv run pytest -q

lint:  ## Style and static checks
	uv run ruff check .
	uv run ruff format --check .

typecheck:  ## Basic type checking
	uv run mypy

eval:  ## Retrieval and answer metrics (added in Phase 2)
	@echo "not implemented yet: Phase 2"

bench:  ## Show logged benchmark runs (added in Phase 6)
	@echo "not implemented yet: Phase 6"

demo:  ## Run the demo (added in Phase 8)
	@echo "not implemented yet: Phase 8"
