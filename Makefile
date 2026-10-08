.PHONY: lint test e2e

lint:
	ruff check .
	ruff format --check .
	mypy

test:
	pytest -q

e2e:
	pytest -q -m e2e tests/e2e
