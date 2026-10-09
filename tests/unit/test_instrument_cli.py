from typing import Any

from typer.testing import CliRunner

from agentqa.cli import app

runner = CliRunner()


def test_instrument_configures_environment_and_propagates_child_exit(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 12345

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            captured["command"] = command
            captured.update(kwargs)

        def wait(self) -> int:
            return 7

        def poll(self) -> None:
            return None

    monkeypatch.setattr("agentqa.cli.instrument._launcher", lambda: "/bin/otel")
    monkeypatch.setattr(
        "agentqa.cli.instrument._installed_instrumentors",
        lambda: {"openai", "langchain", "httpx"},
    )
    monkeypatch.setattr("agentqa.cli.instrument.subprocess.Popen", FakeProcess)

    result = runner.invoke(
        app,
        [
            "instrument",
            "--framework",
            "openai",
            "--exporter",
            "http://127.0.0.1:9999",
            "--",
            "python",
            "agent.py",
        ],
    )

    assert result.exit_code == 7
    assert captured["command"] == ["/bin/otel", "python", "agent.py"]
    assert captured["start_new_session"] is True
    environment = captured["env"]
    assert environment["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:9999"
    assert environment["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    assert environment["OPENINFERENCE_HIDE_INPUTS"] == "true"
    assert "langchain" in environment["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"]


def test_instrument_reports_optional_dependency_hint(monkeypatch: Any) -> None:
    monkeypatch.setattr("agentqa.cli.instrument._installed_instrumentors", set)

    result = runner.invoke(
        app,
        ["instrument", "--framework", "openai", "--", "python", "agent.py"],
    )

    assert result.exit_code == 2
    assert "agentqakit[instrumentation]" in result.stderr
