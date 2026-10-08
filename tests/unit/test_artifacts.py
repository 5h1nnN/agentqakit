import json
from pathlib import Path

import pytest

from agentqa.conventions import new_id
from agentqa.storage import ARTIFACT_SCHEMA_VERSION, ArtifactError, ArtifactWriter


def test_creates_layout_redacts_secrets_and_verifies_checksums(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path / "artifacts", new_id())
    root = writer.initialize({"status": "CREATED"})

    writer.write_resolved_target(
        {
            "type": "http",
            "headers": {
                "authorization": "Bearer top-secret",
                "x-request-id": "safe",
            },
        }
    )
    writer.append_result({"test_case_id": "case-1", "passed": True, "apiKey": "secret"})

    assert (root / "traces").is_dir()
    assert (root / "cassettes").is_dir()
    assert (root / "failures").is_dir()
    assert writer.verify_checksum("run.json")
    assert writer.verify_checksum("resolved-target.json")
    assert writer.verify_checksum("results.jsonl")

    target = json.loads((root / "resolved-target.json").read_text(encoding="utf-8"))
    assert target["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert target["headers"]["authorization"] == "***"
    assert target["headers"]["x-request-id"] == "safe"
    assert writer.read_result_records() == [
        {"apiKey": "***", "passed": True, "test_case_id": "case-1"}
    ]


def test_summary_finalizes_writer(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, new_id())
    writer.initialize()

    writer.finalize_summary({"total": 0})

    assert writer.verify_checksum("summary.json")
    with pytest.raises(ArtifactError, match="already finalized"):
        writer.append_result({"passed": True})


def test_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="one non-empty path component"):
        ArtifactWriter(tmp_path, "../outside")

    writer = ArtifactWriter(tmp_path, new_id())
    writer.initialize()
    with pytest.raises(ArtifactError, match="trace_id"):
        writer.write_trace("../outside", {"events": []})


def test_failed_atomic_replace_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = ArtifactWriter(tmp_path, new_id())
    root = writer.initialize({"status": "CREATED"})
    before = (root / "run.json").read_bytes()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr("agentqa.storage.artifacts.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        writer.write_run({"status": "RUNNING"})

    assert (root / "run.json").read_bytes() == before


def test_detects_tampered_results(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, new_id())
    root = writer.initialize()
    writer.append_result({"passed": True})
    (root / "results.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="checksum mismatch"):
        writer.read_result_records()
