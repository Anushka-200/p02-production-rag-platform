
.PHONY: setup up down migrate test eval eval-fast lint typecheck eval bench demo adversarial demo-tenants

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

eval:  ## Full eval: retrieval modes + answers + judge (about $1.5; add ARGS=--yes to skip the cost prompt)
	uv run python -m evals.run --answers --judge --log $(ARGS)

adversarial: ## isolation attacks: prints "N attacks, 0 leaks"
	uv run pytest tests/test_adversarial.py -q -s | grep -E "ADVERSARIAL|passed|failed"

eval-fast:  ## Retrieval metrics only (free)
	uv run python -m evals.run --label retrieval-only

bench:  ## Show logged benchmark runs (added in Phase 6)
	@echo "not implemented yet: Phase 6"

demo-tenants: ## live two-company run (needs: API with ENVIRONMENT=prod on 8001 and the worker running)
	uv run python -m scripts.two_tenant_demo --yes

demo:  ## Run the demo (added in Phase 8)
	@echo "not implemented yet: Phase 8"
