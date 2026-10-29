.PHONY: up down test lint format migrate

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
