"""Exercise the T16 collaboration golden path through the official MCP client."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml
from mcp import Client
from mcp.types import TextResourceContents

from agentqa.evaluators import DeepSeekProvider, DeepSeekSettings, load_llm_environment
from agentqa.mcp.server import create_mcp_server
from agentqa.models import AgentTestSpec, GenerationGoal
from agentqa.specs import load_spec_file, load_specs, normalize_spec
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "examples" / "showcase" / "campus_ops_agent"


def _result(value: Any) -> dict[str, Any]:
    content = value.structured_content
    if value.is_error or not isinstance(content, dict):
        detail = value.content[0].text if value.content else "missing MCP response"
        raise RuntimeError(detail)
    return cast(dict[str, Any], content)


def _resource(value: Any) -> dict[str, Any]:
    content = value.contents[0]
    if not isinstance(content, TextResourceContents):
        raise RuntimeError("MCP resource was not text")
    payload = json.loads(content.text)
    if not isinstance(payload, dict):
        raise RuntimeError("MCP resource was not an object")
    return cast(dict[str, Any], payload)


def _portable_golden(workspace: Path) -> None:
    specs = [
        item.spec.model_copy(update={"target_ref": None})
        for item in load_specs(workspace / "agentqa" / "golden-tests")
        if isinstance(item.spec, AgentTestSpec)
    ]
    destination = workspace / "agentqa" / "t16-portable-golden.yaml"
    destination.write_text(
        yaml.safe_dump_all(
            [normalize_spec(spec) for spec in specs],
            allow_unicode=True,
            explicit_start=True,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )


def _bounded_generation_goal(workspace: Path, max_cases: int) -> Path:
    source = workspace / "agentqa" / "generation-goal-comparison.yaml"
    goals = [item.spec for item in load_spec_file(source) if isinstance(item.spec, GenerationGoal)]
    if len(goals) != 1:
        raise ValueError("MCP generation goal is missing")
    goal = goals[0].model_copy(update={"max_cases": max_cases})
    destination = workspace / "agentqa" / "t16-generation-goal.yaml"
    destination.write_text(
        yaml.safe_dump(normalize_spec(goal), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return destination


async def run(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be absent or empty")
    workspace = output / "workspace"
    shutil.copytree(
        SHOWCASE,
        workspace,
        ignore=shutil.ignore_patterns(".pytest_cache", ".ruff_cache", "__pycache__", "artifacts"),
    )
    _portable_golden(workspace)
    generation_goal = _bounded_generation_goal(workspace, args.max_generated_cases)
    load_llm_environment(Path(args.env_file))
    settings = DeepSeekSettings.from_environment()
    settings = replace(
        settings,
        max_tokens=args.llm_max_tokens,
        timeout_seconds=args.llm_timeout_seconds,
    )
    provider = DeepSeekProvider(settings)
    server = create_mcp_server(
        workspace,
        manifest_path="agentqa/manifest-v2.yaml",
        provider=provider,
    )
    goal_matches = [
        item.spec
        for item in load_spec_file(workspace / "agentqa" / "generation-goal-code.yaml")
        if isinstance(item.spec, GenerationGoal)
    ]
    if len(goal_matches) != 1:
        raise ValueError("code-evidence goal is missing")
    code_evidence = [
        item.model_dump(mode="json", by_alias=True) for item in goal_matches[0].code_evidence
    ]
    transcript: list[dict[str, Any]] = []

    async with Client(server, raise_exceptions=True) as client:
        manifest_resource = _resource(await client.read_resource("agentqa://project/manifest"))
        transcript.append({"step": "read_manifest", "result": manifest_resource})
        generated: dict[str, Any] | None = None
        for attempt in range(1, args.generation_attempts + 1):
            try:
                generated = _result(
                    await client.call_tool(
                        "generate_tests",
                        {
                            "manifest_path": "agentqa/manifest-v2.yaml",
                            "goal_path": generation_goal.relative_to(workspace).as_posix(),
                            "mutations_path": "agentqa/mutations.yaml",
                            "code_evidence": code_evidence,
                        },
                    )
                )
            except RuntimeError as exc:
                transcript.append(
                    {
                        "step": "generate_tests_attempt",
                        "attempt": attempt,
                        "status": "FAILED_SAFE",
                        "error": str(exc)[:2_000],
                    }
                )
            else:
                break
        if generated is None:
            raise RuntimeError("MCP generation attempts exhausted")
        transcript.append({"step": "generate_tests", "result": generated})
        validated = _result(
            await client.call_tool(
                "validate_tests", {"tests_path": cast(str, generated["tests_directory"])}
            )
        )
        transcript.append({"step": "validate_tests", "result": validated})
        authorization = {"allow_side_effects": True, "approval_id": args.approval_id}
        broken = _result(
            await client.call_tool(
                "run_tests",
                {
                    "tests_path": "agentqa/t16-portable-golden.yaml",
                    "target_path": "agentqa/targets/bug-authorization.yaml",
                    "manifest_path": "agentqa/manifest-v2.yaml",
                    **authorization,
                },
            )
        )
        broken_result = _result(
            await client.call_tool("get_run_result", {"run_id": broken["run_id"]})
        )
        failure_uri = cast(list[str], broken_result["failure_resource_uris"])[0]
        failure = _resource(await client.read_resource(failure_uri))
        transcript.append({"step": "read_failure", "result": failure})
        trial_id = cast(str, failure["data"]["trial_id"])
        analysis = _result(
            await client.call_tool(
                "analyze_failure",
                {
                    "run_id": broken["run_id"],
                    "trial_id": trial_id,
                    "code_evidence": code_evidence,
                },
            )
        )
        transcript.append({"step": "analyze_failure", "result": analysis})
        fixed = _result(
            await client.call_tool(
                "run_tests",
                {
                    "tests_path": "agentqa/t16-portable-golden.yaml",
                    "target_path": "agentqa/targets/fixed.yaml",
                    "manifest_path": "agentqa/manifest-v2.yaml",
                    **authorization,
                },
            )
        )
        mutation = _result(
            await client.call_tool(
                "run_mutations",
                {
                    "tests_path": "agentqa/mutation-tests",
                    "target_path": "agentqa/targets/fixed.yaml",
                    "mutations_path": "agentqa/mutations.yaml",
                    "manifest_path": "agentqa/manifest-v2.yaml",
                    **authorization,
                },
            )
        )
        comparison = _result(
            await client.call_tool(
                "compare_runs",
                {"baseline_run_id": broken["run_id"], "candidate_run_id": fixed["run_id"]},
            )
        )
        regression_gate = _result(
            await client.call_tool(
                "evaluate_gate",
                {
                    "gate_path": "agentqa/regression-gate.yaml",
                    "baseline_run_id": broken["run_id"],
                    "candidate_run_id": fixed["run_id"],
                },
            )
        )
        mutation_gate = _result(
            await client.call_tool(
                "evaluate_gate",
                {
                    "gate_path": "agentqa/mutation-gate.yaml",
                    "candidate_run_id": mutation["run_id"],
                },
            )
        )
        transcript.extend(
            [
                {"step": "run_broken", "result": broken},
                {"step": "run_fixed_after_repair", "result": fixed},
                {"step": "run_mutations", "result": mutation},
                {"step": "compare_runs", "result": comparison},
                {"step": "regression_gate", "result": regression_gate},
                {"step": "mutation_gate", "result": mutation_gate},
            ]
        )

    payload = (
        canonical_json_bytes(
            {
                "schema_version": "agentqa.t16-mcp-transcript/v1",
                "provider": provider.name,
                "model": provider.model,
                "provider_parameters": dict(provider.parameters),
                "case_budget": args.max_generated_cases,
                "approval_id": args.approval_id,
                "steps": transcript,
            }
        )
        + b"\n"
    )
    destination = output / "mcp-transcript.json"
    destination.write_bytes(payload)
    destination.with_name(f"{destination.name}.sha256").write_text(
        f"{sha256_digest(payload)}\n", encoding="utf-8"
    )
    print(f"T16 MCP path completed: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmark/results/t16-mcp-deepseek-20261014")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--approval-id", default="t16-human-review-20261014")
    parser.add_argument("--max-generated-cases", type=int, default=4, choices=range(1, 51))
    parser.add_argument("--llm-max-tokens", type=int, default=16_384, choices=range(64, 16_385))
    parser.add_argument("--llm-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--generation-attempts", type=int, default=3, choices=range(1, 6))
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
