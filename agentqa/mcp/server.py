"""Official MCP SDK protocol adapter over AgentQAKit's application facade."""

import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from agentqa.errors import AgentQAError
from agentqa.llm import StructuredLLMProvider
from agentqa.mcp.application import AgentQAMCPApplication, RunExecutor
from agentqa.models import CodeEvidence

_LOGGER = logging.getLogger("agentqa.mcp")


def _request_id(ctx: Context[Any, Any]) -> str:
    try:
        return str(ctx.request_id)
    except LookupError:
        return "unavailable"


def _error(exc: Exception) -> ToolError:
    if isinstance(exc, AgentQAError):
        code = exc.code
        message = exc.message
    else:
        code = "MCP_TOOL_ERROR"
        message = f"operation failed: {type(exc).__name__}"
    return ToolError(
        json.dumps(
            {"error": {"code": code, "message": message}},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _log(ctx: Context[Any, Any], tool: str, result: str, run_id: str | None = None) -> None:
    _LOGGER.info(
        "mcp_tool tool=%s request_id=%s run_id=%s result=%s",
        tool,
        _request_id(ctx),
        run_id or "-",
        result,
    )


def create_mcp_server(
    workspace: Path,
    *,
    manifest_path: str = "agent-manifest.yaml",
    provider: StructuredLLMProvider | None = None,
    run_executor: RunExecutor | None = None,
) -> MCPServer:
    """Create one stdio-ready, workspace-confined MCP server."""
    options: dict[str, Any] = {
        "manifest_path": manifest_path,
        "provider": provider,
    }
    if run_executor is not None:
        options["run_executor"] = run_executor
    application = AgentQAMCPApplication(workspace, **options)
    server = MCPServer(
        "agentqakit",
        title="AgentQAKit",
        description="Local AI-agent testing, evidence, and failure-analysis collaboration.",
        instructions=(
            "Read bounded AgentQAKit resources before calling tools. Paths are always "
            "workspace-relative. Never infer that run tools are side-effect free; honor "
            "their explicit approval fields."
        ),
        log_level="WARNING",
    )

    @server.resource(
        "agentqa://project/manifest",
        name="project-manifest",
        description="Validated, redacted AgentManifest for the fixed workspace.",
        mime_type="application/json",
    )
    def project_manifest() -> str:
        return application.project_manifest_resource()

    @server.resource(
        "agentqa://schemas/agent-test",
        name="agent-test-schema",
        description="Current public AgentTest JSON Schema.",
        mime_type="application/json",
    )
    def agent_test_schema() -> str:
        return application.schema_resource("agent-test")

    @server.resource(
        "agentqa://schemas/agent-manifest",
        name="agent-manifest-schema",
        description="Current public AgentManifest JSON Schema.",
        mime_type="application/json",
    )
    def agent_manifest_schema() -> str:
        return application.schema_resource("agent-manifest")

    @server.resource(
        "agentqa://runs/{run_id}/summary",
        name="run-summary",
        description="Verified bounded summary for one completed run.",
        mime_type="application/json",
    )
    def run_summary(run_id: str) -> str:
        return application.run_summary_resource(run_id)

    @server.resource(
        "agentqa://runs/{run_id}/trace-summary",
        name="trace-summary",
        description="Trace counts, statuses, tool names, and errors without full trace content.",
        mime_type="application/json",
    )
    def trace_summary(run_id: str) -> str:
        return application.run_trace_summary_resource(run_id)

    @server.resource(
        "agentqa://runs/{run_id}/failures/{failure_id}",
        name="failure-summary",
        description="Verified reproduction-bundle summary for one failed trial.",
        mime_type="application/json",
    )
    def failure(run_id: str, failure_id: str) -> str:
        return application.failure_resource(run_id, failure_id)

    @server.resource(
        "agentqa://comparisons/{comparison_id}",
        name="comparison-report",
        description="Verified checksummed comparison report.",
        mime_type="application/json",
    )
    def comparison(comparison_id: str) -> str:
        return application.comparison_resource(comparison_id)

    @server.resource(
        "agentqa://analyses/{analysis_id}",
        name="analysis-report",
        description="Verified AnalysisReport produced by either Web or MCP.",
        mime_type="application/json",
    )
    def analysis(analysis_id: str) -> str:
        return application.analysis_resource(analysis_id)

    @server.resource(
        "agentqa://workflows/{workflow_id}/handoffs/{handoff_id}",
        name="handoff-bundle",
        description="Bounded external-repair task, identities, and evidence Resource URIs.",
        mime_type="application/json",
    )
    def handoff(workflow_id: str, handoff_id: str) -> str:
        return application.handoff_resource(workflow_id, handoff_id)

    readonly = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
    writes_local = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )
    calls_external = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=False,
        open_world_hint=True,
    )

    @server.tool(annotations=readonly, structured_output=True)
    def validate_manifest(
        ctx: Context[Any, Any],
        manifest_path: str | None = None,
    ) -> dict[str, Any]:
        """Validate one workspace-relative AgentManifest and return bounded diagnostics."""
        try:
            result = application.validate_manifest(manifest_path)
            _log(ctx, "validate_manifest", "ok" if result["valid"] else "invalid")
            return result
        except Exception as exc:
            _log(ctx, "validate_manifest", "error")
            raise _error(exc) from None

    @server.tool(annotations=calls_external, structured_output=True)
    async def generate_tests(
        ctx: Context[Any, Any],
        manifest_path: str | None = None,
        goal_path: str | None = None,
        mutations_path: str | None = None,
        code_evidence: list[CodeEvidence] | None = None,
    ) -> dict[str, Any]:
        """Generate a plan and validated test drafts in the restricted draft directory."""
        try:
            result = await application.generate_tests(
                manifest_path=manifest_path,
                goal_path=goal_path,
                mutations_path=mutations_path,
                code_evidence=code_evidence or (),
            )
            _log(ctx, "generate_tests", "ok")
            return result
        except Exception as exc:
            _log(ctx, "generate_tests", "error")
            raise _error(exc) from None

    @server.tool(annotations=readonly, structured_output=True)
    def validate_tests(ctx: Context[Any, Any], tests_path: str) -> dict[str, Any]:
        """Validate workspace-relative AgentTest specs without executing an agent."""
        try:
            result = application.validate_tests(tests_path)
            _log(ctx, "validate_tests", "ok" if result["valid"] else "invalid")
            return result
        except Exception as exc:
            _log(ctx, "validate_tests", "error")
            raise _error(exc) from None

    @server.tool(annotations=calls_external, structured_output=True)
    async def run_tests(
        ctx: Context[Any, Any],
        tests_path: str,
        target_path: str,
        manifest_path: str | None = None,
        allow_side_effects: bool = False,
        approval_id: str | None = None,
        random_seed: int = 0,
        concurrency: int = 1,
    ) -> dict[str, Any]:
        """Run tests through the shared runner after explicit side-effect authorization."""
        try:
            result = await application.run_tests(
                tests_path=tests_path,
                target_path=target_path,
                manifest_path=manifest_path,
                allow_side_effects=allow_side_effects,
                approval_id=approval_id,
                random_seed=random_seed,
                concurrency=concurrency,
            )
            _log(ctx, "run_tests", "ok", cast_str(result.get("run_id")))
            return result
        except Exception as exc:
            _log(ctx, "run_tests", "error")
            raise _error(exc) from None

    @server.tool(annotations=calls_external, structured_output=True)
    async def run_mutations(
        ctx: Context[Any, Any],
        tests_path: str,
        target_path: str,
        mutations_path: str,
        manifest_path: str | None = None,
        allow_side_effects: bool = False,
        approval_id: str | None = None,
        random_seed: int = 0,
        concurrency: int = 1,
    ) -> dict[str, Any]:
        """Run a declared mutation suite through the shared runner."""
        try:
            result = await application.run_mutations(
                tests_path=tests_path,
                target_path=target_path,
                mutations_path=mutations_path,
                manifest_path=manifest_path,
                allow_side_effects=allow_side_effects,
                approval_id=approval_id,
                random_seed=random_seed,
                concurrency=concurrency,
            )
            _log(ctx, "run_mutations", "ok", cast_str(result.get("run_id")))
            return result
        except Exception as exc:
            _log(ctx, "run_mutations", "error")
            raise _error(exc) from None

    @server.tool(annotations=readonly, structured_output=True)
    def get_run_result(ctx: Context[Any, Any], run_id: str) -> dict[str, Any]:
        """Return a small verified run summary and follow-up Resource URIs."""
        try:
            result = application.get_run_result(run_id)
            _log(ctx, "get_run_result", "ok", run_id)
            return result
        except Exception as exc:
            _log(ctx, "get_run_result", "error", run_id)
            raise _error(exc) from None

    @server.tool(annotations=calls_external, structured_output=True)
    async def analyze_failure(
        ctx: Context[Any, Any],
        run_id: str,
        trial_id: str,
        code_evidence: list[CodeEvidence] | None = None,
    ) -> dict[str, Any]:
        """Create a separate evidence-citing AnalysisReport for one failed trial."""
        try:
            result = await application.analyze_failure(
                run_id=run_id,
                trial_id=trial_id,
                code_evidence=code_evidence or (),
            )
            _log(ctx, "analyze_failure", "ok", run_id)
            return result
        except Exception as exc:
            _log(ctx, "analyze_failure", "error", run_id)
            raise _error(exc) from None

    @server.tool(annotations=writes_local, structured_output=True)
    def compare_runs(
        ctx: Context[Any, Any],
        baseline_run_id: str,
        candidate_run_id: str,
    ) -> dict[str, Any]:
        """Compare two verified runs and write one checksummed comparison report."""
        try:
            result = application.compare_runs(baseline_run_id, candidate_run_id)
            _log(ctx, "compare_runs", "ok", candidate_run_id)
            return result
        except Exception as exc:
            _log(ctx, "compare_runs", "error", candidate_run_id)
            raise _error(exc) from None

    @server.tool(annotations=writes_local, structured_output=True)
    def evaluate_gate(
        ctx: Context[Any, Any],
        gate_path: str,
        candidate_run_id: str,
        baseline_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate an existing QualityGate and write one checksummed result."""
        try:
            result = application.evaluate_gate(
                gate_path=gate_path,
                candidate_run_id=candidate_run_id,
                baseline_run_id=baseline_run_id,
            )
            _log(ctx, "evaluate_gate", "ok", candidate_run_id)
            return result
        except Exception as exc:
            _log(ctx, "evaluate_gate", "error", candidate_run_id)
            raise _error(exc) from None

    return server


def cast_str(value: Any) -> str | None:
    """Narrow an optional run identifier without stringifying arbitrary objects."""
    return value if isinstance(value, str) else None


__all__ = ["create_mcp_server"]
