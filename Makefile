
.PHONY: setup up down migrate test eval eval-fast lint typecheck eval bench demo adversarial demo-tenants demo-widget test-widget-js

setup:  ## Install dependencies
	uv sync

up:  ## Start only Postgres (pgvector) and Redis, for development outside Docker
	docker compose up -d postgres redis

up-all:  ## Build and start everything: Postgres, Redis, migrations, API (:8001), worker, UI (:8502)
	docker compose up -d --build --wait

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

bench:  ## List every logged measurement (details in docs/benchmarks.md)
	@grep -E "^## B-" docs/benchmarks.md

eval-ci:  ## The 15-question mocked eval that CI runs (free, needs `make up`)
	uv run pytest tests/test_mini_eval.py -q -s | grep -E "MINI EVAL|passed|failed"

demo-widget: ## serve the demo customer website on :8080 (open http://localhost:8080/?key=YOUR_SITE_KEY)
	cd widget/demo && python3 -m http.server 8080

test-widget-js: ## the widget's JavaScript tests (Node)
	node --test widget/test/parse.test.js

demo-tenants: ## live two-company run (needs: API with ENVIRONMENT=prod on 8001 and the worker running)
	uv run python -m scripts.two_tenant_demo --yes

demo: up-all  ## Start everything and create a demo admin (demo@demo.example / the password printed below)
	@docker compose exec -T -e USER_PASSWORD=demo-password-123 api \
		python -m scripts.create_user --org "Demo Co" --email demo@demo.example --role admin
	@echo ""
	@echo "UI:  http://localhost:8502   login demo@demo.example / demo-password-123"
	@echo "API: http://localhost:8001/docs"
	@echo "Next: follow docs/DEMO_SCRIPT.md (upload a document, ask a question, see the citations)"

purge:  ## Delete expired visitor data (chats, leads, WhatsApp routing); see docs/DATA_POLICY.md
	uv run python -m scripts.purge_expired

redteam:  ## Red-team campaign: 83 attacks against scratch tenants (about $0.11)
	PYTHONPATH=. uv run python -m evals.redteam --yes

api:  ## Run the API on :8001 outside Docker (production mode: the dev login header is off)
	ENVIRONMENT=prod uv run uvicorn app.api.main:_default_app --factory --port 8001

worker:  ## Run the ingestion worker outside Docker
	ENVIRONMENT=prod uv run python -m app.worker

ui:  ## Run the Streamlit UI on :8502 outside Docker (needs the API)
	PYTHONPATH=. API_URL=http://localhost:8001 uv run streamlit run ui/Chat.py --server.port 8502

demo-reset:  ## Reset the demo accounts (removes the sample upload, chats and memories); needs `make api`
	uv run python -m scripts.demo_reset
