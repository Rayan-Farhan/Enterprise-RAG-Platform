.PHONY: help up down restart ps logs lint format typecheck test test-unit test-integration \
        test-e2e smoke migrate migration hooks verify clean \
        eval eval-diff eval-gate eval-validate eval-validate-schema eval-list \
        eval-export-review eval-import-review

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
	@echo "make eval             - run an experiment (CONFIG=name SPLIT=dev)"
	@echo "make eval-diff        - compare two experiments (RUN_A=... RUN_B=...)"
	@echo "make eval-gate        - fail on regression beyond tolerance (RUN_A=... RUN_B=...)"
	@echo "make eval-validate    - validate a golden dataset split against the corpus (SPLIT=dev)"
	@echo "make eval-validate-schema - schema-only dataset checks (no corpus, no database)"
	@echo "make eval-list        - list committed experiments"
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

# Evaluation harness (Stage 4). Every target shells out to app.evaluation.cli,
# which exits non-zero on failure so `eval-gate` can fail a CI job.
SPLIT ?= dev
DATASET_VERSION ?= v1
TOLERANCE ?=
# The gate's defaults are what CI runs: the committed baseline against whichever
# experiment the branch committed most recently.
RUN_A ?= experiment-001-baseline
RUN_B ?= latest

eval:
	python -m app.evaluation.cli run --name "$(CONFIG)" --split $(SPLIT) \
		--dataset-version $(DATASET_VERSION) $(EVAL_ARGS)

eval-diff:
	python -m app.evaluation.cli diff --baseline "$(RUN_A)" --candidate "$(RUN_B)"

eval-gate:
	python -m app.evaluation.cli gate --baseline "$(RUN_A)" --candidate "$(RUN_B)" \
		$(if $(TOLERANCE),--tolerance $(TOLERANCE),)

eval-validate:
	python -m app.evaluation.cli validate --split $(SPLIT) --dataset-version $(DATASET_VERSION)

# Schema and type-coverage checks only, with no corpus and no database. This is
# the form CI can always run; `eval-validate` additionally resolves every
# evidence pointer against PostgreSQL.
eval-validate-schema:
	python -m app.evaluation.cli validate --split dev --no-corpus
	python -m app.evaluation.cli validate --split validation --no-corpus

eval-list:
	python -m app.evaluation.cli list

eval-export-review:
	python -m app.evaluation.cli export-review --run "$(RUN)" --out "$(OUT)"

eval-import-review:
	python -m app.evaluation.cli import-review --file "$(FILE)" --run "$(RUN)"

# The stage exit gate in one command.
verify: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
