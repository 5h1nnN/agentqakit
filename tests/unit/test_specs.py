from pathlib import Path
from textwrap import dedent

import pytest

from agentqa.models import AgentTestSpec, HttpTargetSpec, PythonTargetSpec
from agentqa.specs import (
    SpecLoadError,
    load_spec_file,
    load_specs,
    loaded_collection_hash,
    normalize_spec,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(dedent(content).lstrip(), encoding="utf-8")
    return path


def _agent_test(*, evaluator: str = "schema", budget: int = 1000, extra: str = "") -> str:
    return f"""
    apiVersion: agentqa.dev/v1alpha1
    kind: AgentTest
    metadata:
      name: basic-test
    scenario:
      messages:
        - role: user
          content: hello
    expect:
      output:
        requireNonEmpty: true
      budgets:
        maxLatencyMs: {budget}
    evaluators:
      - type: {evaluator}
    trials: 1
    {extra}
    """


def test_loads_valid_http_and_python_targets(tmp_path: Path) -> None:
    http_path = _write(
        tmp_path / "http.yaml",
        """
        apiVersion: agentqa.dev/v1alpha1
        kind: Target
        metadata:
          name: http-agent
        type: http
        url: http://localhost:8000/chat
        request:
          json:
            message: "{{ input.message }}"
        """,
    )
    python_path = _write(
        tmp_path / "python.yaml",
        """
        apiVersion: agentqa.dev/v1alpha1
        kind: Target
        metadata:
          name: python-agent
        type: python
        entrypoint: examples.agent:run
        """,
    )

    http = load_spec_file(http_path)[0]
    python = load_spec_file(python_path)[0]

    assert isinstance(http.spec, HttpTargetSpec)
    assert str(http.spec.url) == "http://localhost:8000/chat"
    assert isinstance(python.spec, PythonTargetSpec)
    assert python.spec.entrypoint == "examples.agent:run"


@pytest.mark.parametrize(
    ("content", "expected_path"),
    [
        (
            """
            apiVersion: agentqa.dev/v1alpha1
            kind: AgentTest
            metadata:
              labels: {}
            scenario:
              messages:
                - role: user
                  content: hello
            expect:
              output:
                requireNonEmpty: true
            evaluators:
              - type: schema
            """,
            ("metadata", "name"),
        ),
        (_agent_test(budget=0), ("expect", "budgets", "maxLatencyMs")),
        (_agent_test(evaluator="invented"), ("evaluators", 0, "type")),
        (_agent_test(extra="trailz: 2"), ("trailz",)),
    ],
)
def test_reports_strict_validation_errors(
    tmp_path: Path,
    content: str,
    expected_path: tuple[str | int, ...],
) -> None:
    path = _write(tmp_path / "invalid.yaml", content)

    with pytest.raises(SpecLoadError) as raised:
        load_spec_file(path)

    issue = raised.value.issues[0]
    assert issue.path == expected_path
    assert issue.position is not None
    assert issue.position.line >= 1


def test_requires_target_specific_fields(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "target.yaml",
        """
        apiVersion: agentqa.dev/v1alpha1
        kind: Target
        metadata:
          name: broken-http
        type: http
        """,
    )

    with pytest.raises(SpecLoadError) as raised:
        load_spec_file(path)

    assert raised.value.issues[0].path[-1] == "url"


def test_loads_multiple_yaml_documents(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "multiple.yaml",
        """
        apiVersion: agentqa.dev/v1alpha1
        kind: Target
        metadata:
          name: first
        type: python
        entrypoint: example:first
        ---
        apiVersion: agentqa.dev/v1alpha1
        kind: Target
        metadata:
          name: second
        type: python
        entrypoint: example:second
        """,
    )

    loaded = load_spec_file(path)

    assert [item.document_index for item in loaded] == [1, 2]
    assert [item.spec.metadata.name for item in loaded] == ["first", "second"]


def test_directory_order_and_collection_hash_are_stable(tmp_path: Path) -> None:
    _write(tmp_path / "b.yaml", _agent_test().replace("basic-test", "test-b"))
    _write(tmp_path / "a.yaml", _agent_test().replace("basic-test", "test-a"))

    first = load_specs(tmp_path)
    second = load_specs(tmp_path)

    assert [item.source.name for item in first] == ["a.yaml", "b.yaml"]
    assert [item.spec_hash for item in first] == [item.spec_hash for item in second]
    assert loaded_collection_hash(first) == loaded_collection_hash(second)


def test_semantically_equal_yaml_has_equal_hash(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.yaml", _agent_test())
    second = _write(
        tmp_path / "second.yaml",
        _agent_test().replace("kind: AgentTest", "kind: AgentTest  # same value"),
    )

    assert load_spec_file(first)[0].spec_hash == load_spec_file(second)[0].spec_hash


def test_rejects_duplicate_keys_and_unsupported_versions(tmp_path: Path) -> None:
    duplicate = _write(
        tmp_path / "duplicate.yaml",
        _agent_test() + "trials: 2\n",
    )
    unsupported = _write(
        tmp_path / "unsupported.yaml",
        _agent_test().replace("agentqa.dev/v1alpha1", "agentqa.dev/v9"),
    )

    with pytest.raises(SpecLoadError, match="spec validation failed") as duplicate_error:
        load_spec_file(duplicate)
    with pytest.raises(SpecLoadError) as version_error:
        load_spec_file(unsupported)

    assert duplicate_error.value.issues[0].code == "yaml_error"
    assert version_error.value.issues[0].code == "unsupported_version"


def test_environment_values_are_redacted_and_not_echoed_in_errors(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "environment.yaml",
        """
        apiVersion: agentqa.dev/v1alpha1
        kind: Target
        metadata:
          name: remote-agent
        type: http
        url: "{{ env.AGENT_URL }}"
        request:
          headers:
            authorization: "Bearer {{ env.AGENT_TOKEN }}"
        """,
    )
    secret_url = "not-a-url-secret-value"

    with pytest.raises(SpecLoadError) as raised:
        load_spec_file(
            path,
            environment={"AGENT_URL": secret_url, "AGENT_TOKEN": "top-secret-token"},
        )

    diagnostics = "\n".join(issue.format() for issue in raised.value.issues)
    assert secret_url not in diagnostics
    assert "top-secret-token" not in diagnostics


def test_agent_test_serializes_yaml_aliases(tmp_path: Path) -> None:
    path = _write(tmp_path / "test.yaml", _agent_test())

    spec = load_spec_file(path)[0].spec

    assert isinstance(spec, AgentTestSpec)
    serialized = normalize_spec(spec)
    assert serialized["apiVersion"] == "agentqa.dev/v1alpha1"
    assert serialized["expect"]["budgets"]["maxLatencyMs"] == 1000


def test_successful_environment_rendering_keeps_a_redacted_document(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "environment.yaml",
        """
        apiVersion: agentqa.dev/v1alpha1
        kind: Target
        metadata:
          name: remote-agent
        type: http
        url: "{{ env.AGENT_URL }}"
        request:
          headers:
            authorization: "Bearer {{ env.AGENT_TOKEN }}"
        """,
    )

    loaded = load_spec_file(
        path,
        environment={"AGENT_URL": "https://agent.example/chat", "AGENT_TOKEN": "secret"},
    )[0]

    assert isinstance(loaded.spec, HttpTargetSpec)
    assert loaded.spec.request.headers["authorization"] == "Bearer secret"
    headers = loaded.redacted_document["request"]["headers"]
    assert headers["authorization"] == "Bearer ***"
