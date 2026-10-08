"""Export AgentQAKit's public Pydantic models as JSON Schema."""

from pathlib import Path

from agentqa.specs import export_json_schemas


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    for path in export_json_schemas(repository_root / "schemas"):
        print(path.relative_to(repository_root))


if __name__ == "__main__":
    main()
