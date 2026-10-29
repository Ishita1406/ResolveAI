.PHONY: up down test lint format migrate seed reseed tool-schemas chat

up:
	docker compose up -d --build

down:
	docker compose down

test:
	cd backend && uv run pytest

lint:
	cd backend && uv run ruff check . && uv run ruff format --check .

format:
	cd backend && uv run ruff check --fix . && uv run ruff format .

migrate:
	cd backend && uv run alembic upgrade head

seed:
	cd backend && uv run python -m app.db.seed

reseed:
	cd backend && uv run python -m app.db.seed --reset

tool-schemas:
	cd backend && uv run python -m app.tools.schema > ../docs/tool-schemas.json

chat:
	cd backend && uv run python -m app.agent.cli
