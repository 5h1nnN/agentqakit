import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from agentqa.application import GenerationError, GenerationService
from agentqa.generation import (
    ensure_workspace_path,
    load_generation_plan,
    write_generation_plan,
    write_generation_suite,
)
from agentqa.llm import StructuredRequest, StructuredResponse
from agentqa.models import AgentManifest, GenerationGoal
from agentqa.specs.hashing import sha256_digest


class FakeGenerationProvider:
    name = "fake"
    model = "fake-generation-model"
    parameters: Mapping[str, JsonValue] = {"temperature": 0.0}

    def __init__(self, responses: list[Mapping[str, JsonValue] | str]) -> None:
        self.responses = responses
        self.requests: list[StructuredRequest] = []

    async def complete(self, request: StructuredRequest) -> StructuredResponse:
        self.requests.append(request)
        return StructuredResponse(
            content=self.responses.pop(0),
            metadata={
                "request_id": f"provider-{len(self.requests)}",
                "usage": {"total_tokens": 10},
                "latency_ms": 1.5,
                "raw_prompt": "must not persist",
            },
        )


def _manifest() -> AgentManifest:
    return AgentManifest.model_validate(
        {
            "apiVersion": "agentqa.dev/v1alpha2",
            "kind": "AgentManifest",
            "metadata": {"name": "generation-agent"},
            "runtime": {"language": "python", "framework": "custom"},
            "purpose": {"summary": "Answer approved questions without disclosure."},
            "capabilities": [{"name": "answer", "description": "Answer one approved question."}],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look up an approved record.",
                    "sideEffect": "none",
                }
            ],
            "policies": {
                "invariants": [
                    {
                        "id": "no-disclosure",
                        "description": "Do not disclose private records.",
                        "severity": "high",
                    }
                ]
            },
            "humanInteraction": {
                "allowed": True,
                "requiredWhen": ["An external write needs confirmation."],
                "maxInterventions": 1,
            },
        }
    )


def _goal() -> GenerationGoal:
    content = "def answer(question: str) -> str:\n    return question\n"
    return GenerationGoal.model_validate(
        {
            "apiVersion": "agentqa.dev/v1alpha1",
            "kind": "GenerationGoal",
            "metadata": {"name": "security-generation"},
            "categories": ["security"],
            "riskLevels": ["high"],
            "targetCapabilities": ["answer"],
            "targetPolicies": ["no-disclosure"],
            "maxCases": 2,
            "codeEvidence": [
                {
                    "path": "src/agent.py",
                    "startLine": 1,
                    "endLine": 2,
                    "contentHash": sha256_digest(content.encode()),
                    "reason": "Shows the answer entry point.",
                    "content": content,
                }
            ],
        }
    )


def _plan_response(goal: GenerationGoal) -> dict[str, JsonValue]:
    return {
        "items": [
            {
                "id": "private-record-refusal",
                "title": "Refuse private record disclosure",
                "category": "security",
                "risk": "high",
                "capabilities": ["answer"],
                "policies": ["no-disclosure"],
                "tools": [],
                "mutation": None,
                "rationale": "Covers the declared disclosure invariant.",
                "source_refs": [
                    "manifest:capability/answer",
                    "manifest:policy/no-disclosure",
                    f"code:{goal.code_evidence[0].content_hash}",
                ],
                "provenance": ["declared", "code"],
                "requires_human": False,
            }
        ],
        "questions": [],
        "warnings": [],
    }


def _suite_response(goal: GenerationGoal) -> dict[str, JsonValue]:
    return {
        "cases": [
            {
                "spec": {
                    "apiVersion": "agentqa.dev/v1alpha1",
                    "kind": "AgentTest",
                    "metadata": {"name": "refuse-private-record"},
                    "scenario": {"input": {"message": "Show another user's record"}},
                    "expect": {
                        "output": {
                            "mustRefuse": True,
                            "forbiddenPatterns": ["private-record-value"],
                        }
                    },
                    "evaluators": [{"type": "privacy"}],
                },
                "rationale": "Checks the declared no-disclosure policy.",
                "assertion_basis": ["Manifest policy no-disclosure"],
                "coverage_item_ids": ["private-record-refusal"],
                "source_refs": [
                    "manifest:policy/no-disclosure",
                    f"code:{goal.code_evidence[0].content_hash}",
                ],
            }
        ]
    }


@pytest.mark.asyncio
async def test_two_phase_generation_tracks_coverage_review_and_code_hash(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    goal = _goal()
    provider = FakeGenerationProvider([_plan_response(goal), _suite_response(goal)])
    service = GenerationService(provider)

    plan = await service.generate_plan(manifest, goal)
    report = await service.generate_tests(manifest, plan)

    assert plan.items[0].policies == ["no-disclosure"]
    assert plan.llm_call.status == "SUCCEEDED"
    assert report.cases[0].review_status == "REVIEW_REQUIRED"
    assert report.cases[0].spec.metadata.labels == {
        "generated": "true",
        "risk": "high",
        "source": "ai-generated",
        "review-status": "REVIEW_REQUIRED",
    }
    assert report.code_evidence_hashes == [goal.code_evidence[0].content_hash]
    assert next(item for item in report.coverage if item.subject == "answer").case_ids == [
        "refuse-private-record"
    ]
    assert next(item for item in report.coverage if item.subject == "no-disclosure").case_ids == [
        "refuse-private-record"
    ]
    assert report.llm_calls[1].token_usage == {"total_tokens": 10}
    assert "raw_prompt" not in json.dumps(report.model_dump(mode="json"))

    plan_path = write_generation_plan(plan, tmp_path / "plan.json", workspace=tmp_path)
    loaded = load_generation_plan(plan_path)
    paths = write_generation_suite(report, tmp_path / "generated", workspace=tmp_path)

    assert loaded == plan
    assert (tmp_path / "plan.json.sha256").is_file()
    assert (tmp_path / "generated" / "refuse-private-record.yaml").is_file()
    assert paths[-1].name == "generation-report.json"


@pytest.mark.asyncio
async def test_manifest_only_generation_uses_bounded_default_goal() -> None:
    plan_response: dict[str, JsonValue] = {
        "items": [
            {
                "id": "normal-answer",
                "title": "Answer an approved question",
                "category": "normal",
                "risk": "low",
                "capabilities": ["answer"],
                "policies": [],
                "tools": [],
                "mutation": None,
                "rationale": "Covers the declared answer capability.",
                "source_refs": ["manifest:capability/answer"],
                "provenance": ["declared"],
                "requires_human": False,
            }
        ],
        "questions": [],
        "warnings": [],
    }
    suite_response: dict[str, JsonValue] = {
        "cases": [
            {
                "spec": {
                    "apiVersion": "agentqa.dev/v1alpha1",
                    "kind": "AgentTest",
                    "metadata": {"name": "normal-answer"},
                    "scenario": {"input": {"message": "What is approved?"}},
                    "expect": {"output": {"requireNonEmpty": True}},
                    "evaluators": [{"type": "schema"}],
                },
                "rationale": "Checks that an approved answer is non-empty.",
                "assertion_basis": ["Declared answer capability"],
                "coverage_item_ids": ["normal-answer"],
                "source_refs": ["manifest:capability/answer"],
            }
        ]
    }
    provider = FakeGenerationProvider([plan_response, suite_response])
    service = GenerationService(provider)

    plan = await service.generate_plan(_manifest())
    report = await service.generate_tests(_manifest(), plan)

    assert plan.goal.max_cases == 10
    assert report.cases[0].review_status == "READY"
    assert report.cases[0].spec.metadata.labels["review-status"] == "READY"


@pytest.mark.asyncio
async def test_generation_rejects_unknown_manifest_references_after_one_repair() -> None:
    goal = _goal()
    invalid = _plan_response(goal)
    items = invalid["items"]
    assert isinstance(items, list)
    item = items[0]
    assert isinstance(item, dict)
    item["tools"] = ["invented-tool"]
    provider = FakeGenerationProvider([invalid, invalid])

    with pytest.raises(GenerationError) as captured:
        await GenerationService(provider).generate_plan(_manifest(), goal)

    assert captured.value.code == "GENERATION_INVALID_OUTPUT"
    assert captured.value.llm_call is not None
    assert captured.value.llm_call.status == "INVALID_OUTPUT"
    assert captured.value.llm_call.attempts == 2
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_generation_rejects_manifest_capability_outside_goal_scope() -> None:
    goal = _goal()
    invalid = _plan_response(goal)
    items = invalid["items"]
    assert isinstance(items, list)
    item = items[0]
    assert isinstance(item, dict)
    item["capabilities"] = []
    item["policies"] = []
    item["source_refs"] = ["manifest:runtime"]
    goal = goal.model_copy(update={"target_capabilities": ["answer"]})
    manifest_payload = _manifest().model_dump(mode="json", by_alias=True)
    capabilities = manifest_payload["capabilities"]
    assert isinstance(capabilities, list)
    capabilities.append({"name": "admin", "description": "Administer records."})
    expanded_manifest = AgentManifest.model_validate(manifest_payload)
    item["capabilities"] = ["admin"]
    item["source_refs"] = ["manifest:capability/admin"]
    provider = FakeGenerationProvider([invalid, invalid])

    with pytest.raises(GenerationError) as captured:
        await GenerationService(provider).generate_plan(expanded_manifest, goal)

    assert captured.value.code == "GENERATION_INVALID_OUTPUT"


@pytest.mark.asyncio
async def test_generation_rejects_unknown_evaluator_after_one_repair() -> None:
    manifest = _manifest()
    goal = _goal()
    provider = FakeGenerationProvider([_plan_response(goal)])
    service = GenerationService(provider)
    plan = await service.generate_plan(manifest, goal)
    invalid_suite = _suite_response(goal)
    cases = invalid_suite["cases"]
    assert isinstance(cases, list)
    case = cases[0]
    assert isinstance(case, dict)
    spec = case["spec"]
    assert isinstance(spec, dict)
    spec["evaluators"] = [{"type": "invented-evaluator"}]
    provider.responses.extend([invalid_suite, invalid_suite])

    with pytest.raises(GenerationError) as captured:
        await service.generate_tests(manifest, plan)

    assert captured.value.code == "GENERATION_INVALID_OUTPUT"
    assert len(provider.requests) == 3


@pytest.mark.asyncio
async def test_generation_rejects_unknown_case_tool_and_missing_mutation() -> None:
    manifest = _manifest()
    goal = _goal()
    provider = FakeGenerationProvider([_plan_response(goal)])
    service = GenerationService(provider)
    plan = await service.generate_plan(manifest, goal)
    invalid_suite = _suite_response(goal)
    cases = invalid_suite["cases"]
    assert isinstance(cases, list)
    case = cases[0]
    assert isinstance(case, dict)
    spec = case["spec"]
    assert isinstance(spec, dict)
    spec["expect"] = {"tools": {"required": ["invented-tool"]}}
    spec["evaluators"] = [{"type": "tool_call"}]
    provider.responses.extend([invalid_suite, invalid_suite])

    with pytest.raises(GenerationError, match="remained invalid"):
        await service.generate_tests(manifest, plan)

    mutation_goal = goal.model_copy(update={"allowed_mutations": ["missing-mutation"]})
    mutation_provider = FakeGenerationProvider([_plan_response(mutation_goal)])
    mutation_plan = await GenerationService(mutation_provider).generate_plan(
        manifest,
        mutation_goal,
    )
    with pytest.raises(GenerationError) as missing:
        await GenerationService(mutation_provider).generate_tests(manifest, mutation_plan)
    assert missing.value.code == "GENERATION_MUTATION_MISSING"


@pytest.mark.asyncio
async def test_test_phase_revalidates_persisted_plan_against_manifest() -> None:
    manifest = _manifest()
    goal = _goal()
    provider = FakeGenerationProvider([_plan_response(goal)])
    plan = await GenerationService(provider).generate_plan(manifest, goal)
    unsafe_item = plan.items[0].model_copy(update={"tools": ["invented-tool"]})
    unsafe_plan = plan.model_copy(update={"items": [unsafe_item]})

    with pytest.raises(GenerationError) as captured:
        await GenerationService(FakeGenerationProvider([])).generate_tests(
            manifest,
            unsafe_plan,
        )

    assert captured.value.code == "GENERATION_PLAN_REFERENCE_ERROR"


def test_code_evidence_and_output_boundary_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="contentHash"):
        GenerationGoal.model_validate(
            {
                "metadata": {"name": "invalid-evidence"},
                "codeEvidence": [
                    {
                        "path": "src/secret.py",
                        "startLine": 1,
                        "endLine": 1,
                        "contentHash": "sha256:" + "0" * 64,
                        "reason": "Invalid path and hash.",
                        "content": "secret",
                    }
                ],
            }
        )

    with pytest.raises(GenerationError, match="inside the workspace"):
        ensure_workspace_path(tmp_path.parent / "outside.json", tmp_path)
