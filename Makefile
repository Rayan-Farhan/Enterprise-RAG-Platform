.PHONY: help up down restart ps logs lint format typecheck test test-unit test-integration \
        test-e2e smoke migrate migration hooks verify clean

help:
	@echo "Enterprise Multimodal RAG Platform - Command Center"
	@echo "----------------------------------------------------"
	@echo "make up               - Start Docker Compose services"
	@echo "make down             - Stop and remove Docker Compose containers"
	@echo "make restart          - Restart Docker Compose services"
	@echo "make logs             - Tail Docker Compose logs"
	@echo "make lint             - Run ruff linter checks"
	@echo "make format           - Run ruff code formatting"
	@echo "make typecheck        - Run mypy type checker"
	@echo "make test             - Run all unit and integration tests"
	@echo "make test-unit        - Run unit tests only"
	@echo "make test-integration - Run integration tests only"
	@echo "make test-cov         - Run tests with coverage"
	@echo "make smoke            - Live model gateway smoke test (needs provider API keys)"
	@echo "make migrate          - Apply Alembic migrations to head"
	@echo "make migration        - Autogenerate a new migration (MSG=...)"
	@echo "make hooks            - Install pre-commit hooks"
	@echo "make verify           - lint + typecheck + test (the stage exit gate)"
	@echo "make clean            - Clean build and cache files"

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

ps:
	docker compose ps

logs:
	docker compose logs -f

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy app

test:
	pytest tests

test-unit:
	pytest tests/unit

test-integration:
	pytest tests/integration

test-e2e:
	pytest tests/e2e

test-cov:
	pytest --cov=app --cov-report=term-missing tests

# Live provider calls, excluded from the default suite. Skips per capability when
# the corresponding API key is absent.
smoke:
	pytest -m live tests/e2e

migrate:
	alembic upgrade head

migration:
	alembic revision --autogenerate -m "$(MSG)"

hooks:
	pre-commit install

# The stage exit gate in one command.
verify: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
