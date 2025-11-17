# Makefile
# Makefile for LLM Service
# Provides convenient commands for development and deployment

.PHONY: help install dev test lint format clean docker-build docker-up docker-down docker-logs

# Default target
help:
	@echo "LLM Service - Available Commands"
	@echo "================================="
	@echo "Development:"
	@echo "  make install       - Install dependencies"
	@echo "  make dev          - Run development server"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start services"
	@echo "  make docker-down  - Stop services"
	@echo "  make docker-logs  - View logs"
	@echo "  make docker-shell - Open shell in container"
	@echo ""
	@echo "Ollama:"
	@echo "  make ollama-pull  - Pull Ollama models"
	@echo "  make ollama-list  - List Ollama models"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Clean temporary files"
	@echo "  make clean-models - Clean downloaded models"

# Development
install:
	pip install -r requirements.txt

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=app --cov-report=html --cov-report=term

lint:
	flake8 app/ tests/
	mypy app/

format:
	black app/ tests/
	isort app/ tests/

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose exec llm-service /bin/bash

docker-restart:
	docker-compose restart

# Ollama commands
ollama-pull:
	@echo "Pulling Ollama models..."
	docker exec -it ollama ollama pull llama2
	docker exec -it ollama ollama pull mistral
	docker exec -it ollama ollama pull codellama

ollama-list:
	docker exec -it ollama ollama list

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

clean-models:
	@echo "Warning: This will delete all downloaded models!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/huggingface/*; \
		rm -rf models/fine-tuned/*; \
		echo "Models cleaned"; \
	fi

# Database migrations (if using database)
migrate:
	alembic upgrade head

migrate-create:
	@read -p "Migration name: " name; \
	alembic revision --autogenerate -m "$$name"

# Production
prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Documentation
docs:
	@echo "Opening API documentation..."
	@echo "Swagger UI: http://localhost:8000/docs"
	@echo "ReDoc: http://localhost:8000/redoc"