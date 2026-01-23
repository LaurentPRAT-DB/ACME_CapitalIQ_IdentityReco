.PHONY: help install install-dev test lint format clean run-example

help:
	@echo "Available commands:"
	@echo "  make install       - Install package dependencies with uv"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make test          - Run tests with pytest"
	@echo "  make lint          - Run linting with ruff"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make run-example   - Run quick example"

install:
	uv pip install -r requirements.txt

install-dev:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/

format:
	black src/ tests/ notebooks/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-example:
	python example.py
