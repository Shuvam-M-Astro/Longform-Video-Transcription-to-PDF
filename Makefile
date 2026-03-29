# Makefile for Video Processing System

.PHONY: help install install-dev test test-cov lint format clean run dev setup db-init db-migrate

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run test suite"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting tools"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean up temporary files"
	@echo "  run          Run the web application"
	@echo "  dev          Run in development mode"
	@echo "  setup        Full development setup"
	@echo "  db-init      Initialize database"
	@echo "  db-migrate   Run database migrations"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing tests/

# Code Quality
lint:
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Running
run:
	python web_app.py

dev:
	FLASK_ENV=development FLASK_DEBUG=True python web_app.py

# Setup
setup: install-dev
	pre-commit install
	cp .env.example .env
	@echo "Please edit .env with your configuration"

db-init:
	python init_database.py

db-migrate:
	alembic upgrade head