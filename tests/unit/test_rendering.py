from typing import Any

import pytest

from agentqa.specs import TemplateRenderError, render_templates


def test_full_input_token_preserves_the_original_type() -> None:
    rendered = render_templates("{{ input.retry_count }}", input_values={"retry_count": 3})

    assert rendered.value == 3
    assert rendered.redacted == 3


def test_embedded_environment_token_has_a_redacted_view() -> None:
    rendered = render_templates(
        {"authorization": "Bearer {{ env.AGENT_TOKEN }}"},
        environment={"AGENT_TOKEN": "secret-value"},
    )

    assert rendered.value == {"authorization": "Bearer secret-value"}
    assert rendered.redacted == {"authorization": "Bearer ***"}


def test_can_leave_input_tokens_for_runtime_rendering() -> None:
    rendered = render_templates(
        {"message": "prefix {{ input.message }}"},
        resolve_input=False,
    )

    assert rendered.value == {"message": "prefix {{ input.message }}"}


@pytest.mark.parametrize(
    "template",
    [
        "{{ input.missing }}",
        "{{ env.MISSING }}",
        "{{ __import__('os').system('echo unsafe') }}",
        "{{ input.value | upper }}",
    ],
)
def test_missing_or_executable_expressions_fail_closed(template: str) -> None:
    with pytest.raises(TemplateRenderError):
        render_templates(template, input_values={}, environment={})


def test_reference_cycles_fail_closed() -> None:
    cyclic: list[Any] = []
    cyclic.append(cyclic)

    with pytest.raises(TemplateRenderError, match="reference cycle"):
        render_templates(cyclic)
