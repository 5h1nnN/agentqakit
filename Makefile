.PHONY: lint test e2e web

lint:
	ruff check .
	ruff format --check .
	mypy

test:
	python -m pytest -q

e2e:
	python -m pytest -q -m e2e tests/e2e

web:
	cd apps/web && npm ci && npm test && npm run build
