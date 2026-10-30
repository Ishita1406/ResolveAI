.PHONY: up down test lint format migrate seed reseed tool-schemas chat ingest-policies eval-retrieval eval eval-validate eval-smoke eval-adversarial eval-gate voice-replay voice-fixtures

up:
	docker compose up -d --build

down:
	docker compose down

test:
	cd backend && uv run pytest

lint:
	cd backend && uv run ruff check . ../evals && uv run ruff format --check . ../evals

format:
	cd backend && uv run ruff check --fix . ../evals && uv run ruff format . ../evals

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

# Embed docs/policies/*.md into policy_chunks (first run downloads the embedding model).
ingest-policies:
	cd backend && uv run python -m app.retrieval.ingest

# Recall@1/3/5 and MRR for vector, full-text and hybrid search; writes the report.
eval-retrieval:
	cd backend && uv run python -m app.retrieval.evaluate --output ../evals/retrieval/results.md

# Agent evaluation: simulated customers over /ws/chat, a fresh database per run.
# Pass options through ARGS, e.g. make eval ARGS="--tags refund --runs 1 --no-judge".
eval:
	cd backend && PYTHONPATH=../evals uv run python -m harness run $(ARGS)

# Check the scenario files against the seed data (no database, no LLM).
eval-validate:
	cd backend && PYTHONPATH=../evals uv run python -m harness validate

# The CI smoke suite (15 scenarios, 1 run, no judge), then the gate against the baseline.
eval-smoke:
	cd backend && PYTHONPATH=../evals uv run python -m harness run --tags smoke --runs 1 --no-judge --output-dir ../evals/results/smoke $(ARGS)

eval-gate:
	cd backend && PYTHONPATH=../evals uv run python -m harness compare --results ../evals/results/smoke/results.json --baseline ../evals/baselines/smoke.json $(ARGS)

# The 60-scenario adversarial suite.
eval-adversarial:
	cd backend && PYTHONPATH=../evals uv run python -m harness run --tags adversarial $(ARGS)

# Voice pipeline on recorded audio (no microphone). FILES: WAVs, one customer turn each.
# Offline by default (fake STT/TTS, echo LLM); ARGS="" with CARTESIA_API_KEY and
# GROQ_API_KEY set runs the real providers. Example:
#   make voice-fixtures && make voice-replay FILES="../voice-fixtures/customer-*.wav" ARGS=""
voice-replay:
	cd backend && uv run python -m app.voice.replay run $(FILES) --out-dir ../voice-replay $(if $(ARGS),$(ARGS),--fake --echo)

# Customer test audio spoken by Cartesia TTS (needs CARTESIA_API_KEY).
voice-fixtures:
	cd backend && uv run python -m app.voice.replay fixtures ../voice-fixtures
