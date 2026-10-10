import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentqa.models.storage import FindingReview
from agentqa.storage import (
    FileResultStore,
    ResultStore,
    ResultStoreError,
    SQLiteResultStore,
)
from tests.fixtures.stored_runs import create_storable_run


def _file_store(path: Path) -> ResultStore:
    return FileResultStore(path / "file-store")


def _sqlite_store(path: Path) -> ResultStore:
    return SQLiteResultStore(path / "results.sqlite3")


@pytest.mark.parametrize("factory", [_file_store, _sqlite_store])
async def test_result_store_contract(
    tmp_path: Path,
    factory: Callable[[Path], ResultStore],
) -> None:
    run, artifacts = create_storable_run(tmp_path / "artifacts", "stored-run")
    store = factory(tmp_path)
    await store.initialize()

    saved = await store.save_run(
        run,
        project_id="campus",
        artifact_path=artifacts,
    )
    again = await store.save_run(
        run,
        project_id="campus",
        artifact_path=artifacts,
    )

    assert saved == again
    assert saved.run.results[0].target_result.output == {
        "answer": "ok",
        "api_key": "***",
    }
    assert await store.get_run(run.run_id) == saved
    summaries = await store.list_runs("campus")
    assert [item.run_id for item in summaries] == [run.run_id]
    assert summaries[0].finding_count == 1
    assert await store.list_results(run.run_id) == saved.run.results

    trace = await store.get_trace(run.results[0].trace_id)
    assert trace is not None
    assert trace.status == "missing"

    findings = await store.list_findings(run.run_id)
    assert len(findings) == 1
    assert findings[0].details == {"api_key": "***"}
    review = FindingReview(
        finding_id=findings[0].finding_id,
        status="confirmed",
        note="reproduced locally",
        updated_at=datetime(2026, 10, 10, tzinfo=UTC),
    )
    await store.save_review(review)
    assert await store.get_review(review.finding_id) == review
    await store.close()


@pytest.mark.parametrize("factory", [_file_store, _sqlite_store])
async def test_result_store_rejects_conflicting_run_ids(
    tmp_path: Path,
    factory: Callable[[Path], ResultStore],
) -> None:
    first, first_artifacts = create_storable_run(
        tmp_path / "first-artifacts", "same-run", passed=True
    )
    conflicting, conflicting_artifacts = create_storable_run(
        tmp_path / "second-artifacts", "same-run", passed=False
    )
    store = factory(tmp_path)
    await store.save_run(first, artifact_path=first_artifacts)

    with pytest.raises(ResultStoreError, match="different evidence"):
        await store.save_run(conflicting, artifact_path=conflicting_artifacts)


@pytest.mark.parametrize("factory", [_file_store, _sqlite_store])
async def test_result_store_supports_failed_run_lifecycle(
    tmp_path: Path,
    factory: Callable[[Path], ResultStore],
) -> None:
    run, artifacts = create_storable_run(
        tmp_path / "artifacts", "failed-run", failed_lifecycle=True
    )
    store = factory(tmp_path)

    stored = await store.save_run(run, artifact_path=artifacts)

    assert stored.run.status == "FAILED"
    assert stored.run.exit_code == 2
    assert (await store.list_runs())[0].status == "FAILED"


@pytest.mark.parametrize("factory", [_file_store, _sqlite_store])
async def test_result_store_rejects_path_like_run_ids(
    tmp_path: Path,
    factory: Callable[[Path], ResultStore],
) -> None:
    run, _ = create_storable_run(tmp_path / "artifacts", "safe-run")
    unsafe = run.model_copy(update={"run_id": "../escape"})

    with pytest.raises(ResultStoreError, match="valid resource name"):
        await factory(tmp_path).save_run(unsafe)


async def test_file_store_detects_record_corruption(tmp_path: Path) -> None:
    run, artifacts = create_storable_run(tmp_path / "artifacts", "corrupt-run")
    store = FileResultStore(tmp_path / "store")
    await store.save_run(run, artifact_path=artifacts)
    record = tmp_path / "store" / "projects" / "default" / "runs" / "corrupt-run.json"
    record.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ResultStoreError, match="checksum mismatch"):
        await store.get_run(run.run_id)


async def test_sqlite_store_creates_required_normalized_tables(tmp_path: Path) -> None:
    store = SQLiteResultStore(tmp_path / "results.sqlite3")
    await store.initialize()
    connection = sqlite3.connect(store.path)
    try:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        connection.close()

    assert {
        "projects",
        "runs",
        "test_results",
        "traces",
        "findings",
        "review_status",
    }.issubset(names)
