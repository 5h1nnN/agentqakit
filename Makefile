.PHONY: lint test e2e web web-e2e schemas leak-scan acceptance

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

web-e2e:
	cd apps/web && npm run test:e2e

schemas:
	python scripts/export_schemas.py

leak-scan:
	python scripts/scan_artifact_leaks.py

acceptance: lint test web schemas leak-scan web-e2e
