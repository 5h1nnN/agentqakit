import json
from pathlib import Path

from agentqa.security import scan_artifact_leaks


def test_leak_scan_accepts_bounded_handoff(tmp_path: Path) -> None:
    path = tmp_path / "handoff-bundle.json"
    path.write_text(
        json.dumps(
            {
                "handoff_resource_uri": "agentqa://workflows/w/handoffs/h",
                "developer_task": "Fix the cited failure and rerun the reviewed tests.",
                "baseline_source_snapshot_hash": "sha256:" + "1" * 64,
            }
        ),
        encoding="utf-8",
    )

    assert scan_artifact_leaks([tmp_path]) == []


def test_leak_scan_reports_secret_and_raw_code_without_echoing_values(tmp_path: Path) -> None:
    path = tmp_path / "workflow-report.json"
    path.write_text(
        json.dumps({"api_key": "sk-abcdefghijklmnopqrstuvwxyz", "content": "private code"}),
        encoding="utf-8",
    )

    findings = scan_artifact_leaks([tmp_path])

    assert {item.code for item in findings} == {
        "OPENAI_STYLE_KEY",
        "OVER_BROAD_EXCHANGE_FIELD",
    }
    assert all("abcdefghijklmnopqrstuvwxyz" not in repr(item) for item in findings)
