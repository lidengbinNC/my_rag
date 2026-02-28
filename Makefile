.PHONY: install dev run test lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

run:
	uvicorn my_rag.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check my_rag/
	ruff format my_rag/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/
