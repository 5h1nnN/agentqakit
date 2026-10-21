.PHONY: lint test e2e web web-e2e schemas leak-scan package release-check acceptance

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
	python scripts/sync_web_assets.py

web-e2e:
	cd apps/web && npm run test:e2e

schemas:
	python scripts/export_schemas.py

leak-scan:
	python scripts/scan_artifact_leaks.py

package: web
	python -m build
	python -m twine check dist/*

release-check: lint test schemas leak-scan package
	python scripts/check_release.py --dist dist

acceptance: lint test web schemas leak-scan web-e2e
