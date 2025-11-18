# Makefile for Digital Twin Factory System

.PHONY: help install test format lint typecheck check clean docker-build docker-up docker-down db-init

# Variables
PYTHON := poetry run python
PYTEST := poetry run pytest
BLACK := poetry run black
RUFF := poetry run ruff
MYPY := poetry run mypy

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install dependencies with Poetry
	poetry install --with dev,test
	poetry run pre-commit install

install-prod:  ## Install production dependencies only
	poetry install --without dev,test

update:  ## Update dependencies
	poetry update

# Development
dev-api:  ## Run API server in development mode
	$(PYTHON) -m uvicorn src.api.main:app --reload --port 8000

dev-frontend:  ## Run frontend development server
	cd frontend && npm run dev

dev-mlflow:  ## Run MLflow tracking server
	poetry run mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 0.0.0.0 --port 5000

dev-workers:  ## Run Celery workers
	poetry run celery -A src.workers.celery_app worker --loglevel=info

# Testing
test:  ## Run all tests
	$(PYTEST) tests/ -v

test-unit:  ## Run unit tests only
	$(PYTEST) tests/unit/ -v

test-integration:  ## Run integration tests
	$(PYTEST) tests/integration/ -v

test-e2e:  ## Run end-to-end tests
	$(PYTEST) tests/e2e/ -v

test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term

test-perf:  ## Run performance tests
	$(PYTEST) tests/performance/ -v

load-test:  ## Run load testing with Locust
	poetry run locust -f tests/performance/locustfile.py \
		--host http://localhost:8000 \
		--users 100 --spawn-rate 10

# Code Quality
format:  ## Format code with Black
	$(BLACK) src/ tests/ scripts/
	cd frontend && npm run format

format-check:  ## Check code formatting
	$(BLACK) --check src/ tests/ scripts/

lint:  ## Lint code with Ruff
	$(RUFF) src/ tests/ scripts/

lint-fix:  ## Auto-fix linting issues
	$(RUFF) --fix src/ tests/ scripts/

typecheck:  ## Type check with MyPy
	$(MYPY) src/

check: format-check lint typecheck test  ## Run all checks (CI pipeline)

# Data Management
data-download:  ## Download datasets
	$(PYTHON) scripts/download_datasets.py

data-preprocess:  ## Preprocess datasets
	$(PYTHON) scripts/preprocess_data.py

data-dvc-pull:  ## Pull data from DVC remote
	poetry run dvc pull

data-dvc-push:  ## Push data to DVC remote
	poetry run dvc push

# Model Training
train-baseline:  ## Train baseline model
	$(PYTHON) scripts/train_baseline.py

train-advanced:  ## Train advanced model with hyperparameter tuning
	$(PYTHON) scripts/train_advanced.py

export-onnx:  ## Export model to ONNX
	$(PYTHON) scripts/export_onnx.py

benchmark-model:  ## Benchmark model performance
	$(PYTHON) scripts/benchmark.py

# Database
db-init:  ## Initialize database schema
	$(PYTHON) scripts/init_db.py

db-migrate:  ## Run database migrations
	poetry run alembic upgrade head

db-migrate-create:  ## Create new migration
	@read -p "Migration message: " msg; \
	poetry run alembic revision --autogenerate -m "$$msg"

db-reset:  ## Reset database (WARNING: destructive)
	@echo "⚠️  This will delete all data. Are you sure? (yes/no)"
	@read confirm; \
	if [ "$$confirm" = "yes" ]; then \
		$(PYTHON) scripts/reset_db.py; \
	fi

# Docker
docker-build:  ## Build Docker images
	docker-compose build

docker-up:  ## Start all services with Docker Compose
	docker-compose up -d

docker-down:  ## Stop all services
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f

docker-ps:  ## List running containers
	docker-compose ps

docker-clean:  ## Remove all containers, networks, volumes
	docker-compose down -v --remove-orphans

docker-prod:  ## Start production stack
	docker-compose -f docker-compose.prod.yml up -d

# Kubernetes
k8s-apply:  ## Apply Kubernetes manifests
	kubectl apply -f deploy/kubernetes/namespace.yaml
	kubectl apply -f deploy/kubernetes/configmap.yaml
	kubectl apply -f deploy/kubernetes/secrets.yaml
	kubectl apply -f deploy/kubernetes/

k8s-delete:  ## Delete Kubernetes resources
	kubectl delete -f deploy/kubernetes/

k8s-logs:  ## View Kubernetes logs
	kubectl logs -f -l app=api -n digital-twin

k8s-port-forward:  ## Port forward to API service
	kubectl port-forward svc/api 8000:8000 -n digital-twin

# Monitoring
monitor-prometheus:  ## Start Prometheus
	docker run -d -p 9090:9090 \
		-v $(PWD)/deploy/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
		prom/prometheus

monitor-grafana:  ## Start Grafana
	docker run -d -p 3001:3000 grafana/grafana

# Utilities
clean:  ## Clean temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

clean-all: clean  ## Clean everything including data and models
	@echo "⚠️  This will delete data/ and models/. Continue? (yes/no)"
	@read confirm; \
	if [ "$$confirm" = "yes" ]; then \
		rm -rf data/raw data/processed; \
		rm -rf models/checkpoints models/onnx models/tensorrt; \
	fi

notebook:  ## Launch Jupyter Lab
	poetry run jupyter lab

shell:  ## Open Poetry shell
	poetry shell

requirements:  ## Export requirements.txt
	poetry export -f requirements.txt --output requirements.txt --without-hashes

# Documentation
docs-serve:  ## Serve documentation locally
	cd docs && poetry run mkdocs serve

docs-build:  ## Build documentation
	cd docs && poetry run mkdocs build

# Git hooks
pre-commit:  ## Run pre-commit hooks manually
	poetry run pre-commit run --all-files

# Quick start
quickstart: install data-download train-baseline  ## Quick start: install, download data, train model
	@echo "✅ Quick start complete!"
	@echo "Run 'make dev-api' to start the API server"
