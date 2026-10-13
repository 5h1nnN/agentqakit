import pytest

from agentqa.llm import StructuredRequest


def test_structured_request_enforces_prompt_attempt_and_schema_bounds() -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        StructuredRequest(
            use_case="test_generation",
            system_prompt="",
            prompt="generate",
            response_schema={"type": "object"},
        )

    with pytest.raises(ValueError, match="attempt"):
        StructuredRequest(
            use_case="failure_analysis",
            system_prompt="Analyze evidence.",
            prompt="evidence",
            response_schema={"type": "object"},
            attempt=3,
        )

    with pytest.raises(ValueError, match="response schema"):
        StructuredRequest(
            use_case="semantic_judge",
            system_prompt="Judge evidence.",
            prompt="evidence",
            response_schema={"description": "x" * (64 * 1024)},
        )

    with pytest.raises(ValueError, match="valid Draft 2020-12"):
        StructuredRequest(
            use_case="semantic_judge",
            system_prompt="Judge evidence.",
            prompt="evidence",
            response_schema={"type": "invented"},
        )

    with pytest.raises(ValueError, match="max_output_tokens"):
        StructuredRequest(
            use_case="test_generation",
            system_prompt="Generate tests.",
            prompt="evidence",
            response_schema={"type": "object"},
            max_output_tokens=32,
        )

    with pytest.raises(ValueError, match="max_retries"):
        StructuredRequest(
            use_case="failure_analysis",
            system_prompt="Analyze evidence.",
            prompt="evidence",
            response_schema={"type": "object"},
            max_retries=2,
        )
