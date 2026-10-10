import json
from pathlib import Path

from agentqa.specs import build_json_schemas, export_json_schemas


def test_checked_in_json_schemas_are_current() -> None:
    schema_directory = Path(__file__).resolve().parents[2] / "schemas"
    expected = build_json_schemas()

    actual = {
        path.name.removesuffix(".schema.json"): json.loads(path.read_text(encoding="utf-8"))
        for path in schema_directory.glob("*.schema.json")
    }

    assert actual == expected


def test_spec_schemas_use_external_camel_case_names() -> None:
    schema = build_json_schemas()["agent-test-spec"]
    properties = schema["properties"]

    assert isinstance(properties, dict)
    assert "apiVersion" in properties
    assert "targetRef" in properties
    assert "api_version" not in properties


def test_storage_schemas_are_part_of_the_public_contract() -> None:
    schemas = build_json_schemas()

    assert {
        "finding-review",
        "stored-finding",
        "stored-run",
        "stored-run-summary",
    }.issubset(schemas)


def test_schema_export_is_deterministic(tmp_path: Path) -> None:
    first = export_json_schemas(tmp_path)
    first_payloads: dict[str, bytes] = {path.name: path.read_bytes() for path in first}
    second = export_json_schemas(tmp_path)
    second_payloads: dict[str, bytes] = {path.name: path.read_bytes() for path in second}

    assert first_payloads == second_payloads
