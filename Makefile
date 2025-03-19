.PHONY: setup test lint format clean build docs

setup:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=videoclipper --cov-report=html

lint:
	flake8 videoclipper tests
	black --check videoclipper tests
	isort --check videoclipper tests
	mypy videoclipper

format:
	black videoclipper tests
	isort videoclipper tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

build:
	python -m build

docs:
	mkdocs build
