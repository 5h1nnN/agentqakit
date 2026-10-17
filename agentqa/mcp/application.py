"""Workspace-confined application facade shared by MCP protocol handlers."""

import json
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar, cast

from pydantic import BaseModel, JsonValue, ValidationError

from agentqa.analysis.io import load_failure_analysis_input, write_analysis_report
from agentqa.application import (
    FailureAnalysisService,
    GenerationService,
    RunExecutionConfig,
    execute_run,
)
from agentqa.comparison import RunArtifact, compare_runs, load_run_artifact, write_report
from agentqa.conventions import new_id
from agentqa.errors import AgentQAError
from agentqa.gates import evaluate_gate
from agentqa.generation import write_generation_plan, write_generation_suite
from agentqa.llm import StructuredLLMProvider
from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    AnalysisReport,
    CodeEvidence,
    ComparisonReport,
    GenerationGoal,
    MutationSpec,
    QualityGateSpec,
    RunResult,
    RunSummary,
    TraceRecord,
)
from agentqa.replay.bundle import extract_failure_bundle
from agentqa.specs import SpecLoadError, hash_spec, load_spec_file, load_specs, normalize_spec
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.specs.schema import build_json_schemas
from agentqa.storage import redact_secrets
from agentqa.traceview import TraceView
from agentqa.workbench.models import HandoffBundle

MCP_RESOURCE_SCHEMA_VERSION = "agentqa.mcp-resource/v1alpha1"
_MAX_RESOURCE_BYTES = 256 * 1024
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_MAX_CASE_PREVIEW = 100
_MAX_TRACE_PREVIEW = 500
_Model = TypeVar("_Model", bound=BaseModel)
RunExecutor = Callable[[RunExecutionConfig], Awaitable[RunResult]]


class MCPApplicationError(AgentQAError):
    """Safe error crossing the AgentQAKit-to-MCP application boundary."""

    def __init__(self, message: str, *, code: str = "MCP_APPLICATION_ERROR") -> None:
        super().__init__(message, code=code)


class WorkspacePaths:
    """Resolve all caller paths under one immutable workspace root."""

    def __init__(self, root: Path) -> None:
        resolved = root.resolve()
        if not resolved.is_dir():
            raise MCPApplicationError(
                "MCP workspace must be an existing directory",
                code="MCP_WORKSPACE_ERROR",
            )
        self.root = resolved

    def resolve(
        self,
        value: str,
        *,
        must_exist: bool = True,
        file_only: bool = False,
    ) -> Path:
        if not value or "\x00" in value or "\\" in value:
            raise MCPApplicationError("workspace path is invalid", code="MCP_PATH_ERROR")
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise MCPApplicationError(
                "workspace path must be relative and cannot contain '..'",
                code="MCP_PATH_ERROR",
            )
        candidate = (self.root / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise MCPApplicationError(
                "workspace path escapes the fixed workspace",
                code="MCP_PATH_ERROR",
            ) from None
        if must_exist and not candidate.exists():
            raise MCPApplicationError("workspace path does not exist", code="MCP_PATH_MISSING")
        if must_exist and file_only and not candidate.is_file():
            raise MCPApplicationError("workspace path must be a file", code="MCP_PATH_ERROR")
        return candidate

    def relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            raise MCPApplicationError(
                "output escaped the fixed workspace",
                code="MCP_PATH_ERROR",
            ) from None

    def component(self, value: str, *, label: str) -> str:
        if (
            not value
            or len(value) > 256
            or value in {".", ".."}
            or PurePosixPath(value).name != value
            or "\\" in value
        ):
            raise MCPApplicationError(
                f"{label} must be one safe path component",
                code="MCP_IDENTIFIER_ERROR",
            )
        return value


def _safe_json(value: Any) -> JsonValue:
    return cast(JsonValue, redact_secrets(value))


def _resource_document(uri: str, data: Any) -> str:
    safe = _safe_json(data)
    checksum = sha256_digest(canonical_json_bytes(safe))
    document = {
        "schema_version": MCP_RESOURCE_SCHEMA_VERSION,
        "uri": uri,
        "checksum": checksum,
        "data": safe,
    }
    payload = canonical_json_bytes(document)
    if len(payload) > _MAX_RESOURCE_BYTES:
        raise MCPApplicationError(
            "MCP resource summary exceeds 262144 bytes",
            code="MCP_RESOURCE_TOO_LARGE",
        )
    return payload.decode("utf-8")


def _resource_error(uri: str, exc: Exception) -> str:
    if isinstance(exc, AgentQAError):
        code = exc.code
        message = exc.message
    else:
        code = "MCP_RESOURCE_ERROR"
        message = f"resource could not be loaded: {type(exc).__name__}"
    return _resource_document(uri, {"error": {"code": code, "message": message}})


def _single_spec(path: Path, expected: type[_Model], label: str) -> _Model:
    loaded = load_spec_file(path)
    matches = [item.spec for item in loaded if isinstance(item.spec, expected)]
    if len(loaded) != 1 or len(matches) != 1:
        raise MCPApplicationError(
            f"{label} path must contain exactly one {expected.__name__}",
            code="MCP_SPEC_SELECTION_ERROR",
        )
    return matches[0]


def _diagnostics(exc: SpecLoadError, workspace: WorkspacePaths) -> list[dict[str, JsonValue]]:
    output: list[dict[str, JsonValue]] = []
    for issue in exc.issues[:100]:
        try:
            source = workspace.relative(issue.source)
        except MCPApplicationError:
            source = issue.source.name
        output.append(
            {
                "source": source,
                "document": issue.document_index,
                "path": [str(item) for item in issue.path],
                "message": issue.message,
                "code": issue.code,
                "line": None if issue.position is None else issue.position.line,
                "column": None if issue.position is None else issue.position.column,
            }
        )
    return output


def _bounded_run_summary(summary: RunSummary) -> dict[str, JsonValue]:
    aggregate = cast(
        dict[str, JsonValue],
        summary.model_dump(mode="json", exclude={"cases"}),
    )
    aggregate["case_count"] = len(summary.cases)
    aggregate["case_preview"] = cast(
        JsonValue,
        [item.model_dump(mode="json") for item in summary.cases[:_MAX_CASE_PREVIEW]],
    )
    aggregate["cases_truncated"] = len(summary.cases) > _MAX_CASE_PREVIEW
    return aggregate


def _tool_run_summary(summary: RunSummary) -> dict[str, JsonValue]:
    aggregate = cast(
        dict[str, JsonValue],
        summary.model_dump(mode="json", exclude={"cases"}),
    )
    aggregate["case_count"] = len(summary.cases)
    return aggregate


class AgentQAMCPApplication:
    """Bounded operations and read-only resources exposed through MCP."""

    def __init__(
        self,
        workspace: Path,
        *,
        manifest_path: str = "agent-manifest.yaml",
        provider: StructuredLLMProvider | None = None,
        run_executor: RunExecutor = execute_run,
    ) -> None:
        self.paths = WorkspacePaths(workspace)
        self.manifest_path = manifest_path
        self.provider = provider
        self._run_executor = run_executor

    @property
    def artifacts(self) -> Path:
        return self.paths.resolve("artifacts", must_exist=False)

    @property
    def reports(self) -> Path:
        return self.paths.resolve(".agentqa/mcp", must_exist=False)

    def _run_directory(self, run_id: str) -> Path:
        return self.artifacts / "runs" / self.paths.component(run_id, label="run_id")

    def _existing_run_directory(self, run_id: str) -> Path:
        directory = self._run_directory(run_id)
        if not directory.is_dir():
            raise MCPApplicationError(
                "run resource was not found",
                code="MCP_RESOURCE_NOT_FOUND",
            )
        return directory

    def _load_run(self, run_id: str) -> RunArtifact:
        return load_run_artifact(self._existing_run_directory(run_id))

    def _require_provider(self) -> StructuredLLMProvider:
        if self.provider is None:
            raise MCPApplicationError(
                "structured LLM provider is not configured",
                code="MCP_LLM_NOT_CONFIGURED",
            )
        return self.provider

    def _manifest(self, path: str | None = None) -> AgentManifest:
        source = self.paths.resolve(path or self.manifest_path, file_only=True)
        return _single_spec(source, AgentManifest, "manifest")

    def project_manifest_resource(self) -> str:
        uri = "agentqa://project/manifest"
        try:
            manifest = self._manifest()
            data = {
                "manifest_hash": hash_spec(manifest),
                "manifest": normalize_spec(manifest),
            }
            return _resource_document(uri, data)
        except Exception as exc:
            return _resource_error(uri, exc)

    def schema_resource(self, name: str) -> str:
        uri = f"agentqa://schemas/{name}"
        schema_names = {"agent-test": "agent-test-spec", "agent-manifest": "agent-manifest"}
        try:
            selected = schema_names.get(name)
            if selected is None:
                raise MCPApplicationError("unknown schema resource", code="MCP_RESOURCE_NOT_FOUND")
            return _resource_document(uri, build_json_schemas()[selected])
        except Exception as exc:
            return _resource_error(uri, exc)

    def run_summary_resource(self, run_id: str) -> str:
        uri = f"agentqa://runs/{run_id}/summary"
        try:
            artifact = self._load_run(run_id)
            return _resource_document(
                uri,
                {
                    "run_id": artifact.run_id,
                    "summary": _bounded_run_summary(artifact.summary),
                    "result_count": len(artifact.results),
                    "mutation_score": (
                        None
                        if artifact.mutation_score is None
                        else artifact.mutation_score.model_dump(mode="json")
                    ),
                },
            )
        except Exception as exc:
            return _resource_error(uri, exc)

    @staticmethod
    def _trace(artifact_root: Path, reader: Any, result: Any) -> TraceRecord | None:
        reference = result.trace_ref
        if reference is None:
            return None
        relative = Path(reference)
        if relative.parent != Path("traces") or relative.name != f"{result.trace_id}.json":
            return None
        if not reader.verify_checksum(relative):
            return None
        try:
            payload = json.loads((artifact_root / relative).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            payload.pop("schema_version", None)
            return TraceRecord.model_validate_json(json.dumps(payload))
        except (OSError, json.JSONDecodeError, ValidationError, TypeError):
            return None

    def run_trace_summary_resource(self, run_id: str) -> str:
        uri = f"agentqa://runs/{run_id}/trace-summary"
        try:
            artifact = self._load_run(run_id)
            traces: list[dict[str, JsonValue]] = []
            for result in artifact.results[:_MAX_TRACE_PREVIEW]:
                trace = self._trace(artifact.root, artifact.reader, result)
                if trace is None:
                    traces.append(
                        {
                            "trial_id": result.trial_id,
                            "trace_id": result.trace_id,
                            "status": "unavailable",
                        }
                    )
                    continue
                view = TraceView(trace)
                tool_names = list(dict.fromkeys(call.name for call in view.tool_calls))[:50]
                tool_names_json: list[JsonValue] = list(tool_names)
                trace_summary: dict[str, JsonValue] = {
                    "trial_id": result.trial_id,
                    "trace_id": trace.trace_id,
                    "status": trace.status,
                    "event_count": len(view.internal_events),
                    "llm_calls": len(view.llm_calls),
                    "tool_calls": len(view.tool_calls),
                    "error_spans": sum(event.status == "ERROR" for event in view.internal_events),
                    "tool_names": tool_names_json,
                    "tool_names_truncated": len(tool_names)
                    < len({call.name for call in view.tool_calls}),
                    "dropped_spans": trace.dropped_spans,
                }
                traces.append(trace_summary)
            return _resource_document(
                uri,
                {
                    "run_id": artifact.run_id,
                    "trace_count": len(artifact.results),
                    "traces": traces,
                    "traces_truncated": len(artifact.results) > _MAX_TRACE_PREVIEW,
                },
            )
        except Exception as exc:
            return _resource_error(uri, exc)

    def failure_resource(self, run_id: str, failure_id: str) -> str:
        uri = f"agentqa://runs/{run_id}/failures/{failure_id}"
        try:
            artifact = self._load_run(run_id)
            selected = [
                result
                for result in artifact.results
                if result.trial_id == failure_id
                or (
                    result.failure_bundle_ref is not None
                    and Path(result.failure_bundle_ref).stem == failure_id
                )
            ]
            if len(selected) != 1 or selected[0].failure_bundle_ref is None:
                raise MCPApplicationError(
                    "failure resource was not found",
                    code="MCP_RESOURCE_NOT_FOUND",
                )
            result = selected[0]
            reference = result.failure_bundle_ref
            assert reference is not None
            relative = Path(reference)
            if relative.parent != Path("failures") or relative.suffix != ".zip":
                raise MCPApplicationError("failure resource path is unsafe", code="MCP_PATH_ERROR")
            if not artifact.reader.verify_checksum(relative):
                raise MCPApplicationError(
                    "failure resource checksum mismatch",
                    code="MCP_RESOURCE_CHECKSUM_ERROR",
                )
            with tempfile.TemporaryDirectory(prefix="agentqa-mcp-bundle-") as directory:
                manifest = extract_failure_bundle(artifact.root / relative, Path(directory))
            return _resource_document(
                uri,
                {
                    "bundle": manifest,
                    "test_case_id": result.test_case_id,
                    "trial_id": result.trial_id,
                    "trace_id": result.trace_id,
                    "passed": result.passed,
                    "trace_status": result.trace_status,
                    "target_status": result.target_result.status,
                    "evaluations": [
                        {
                            "type": evaluation.evaluator_type,
                            "status": evaluation.status,
                            "score": evaluation.score,
                        }
                        for evaluation in result.evaluations
                    ],
                    "violations": [item.model_dump(mode="json") for item in result.violations],
                },
            )
        except Exception as exc:
            return _resource_error(uri, exc)

    def comparison_resource(self, comparison_id: str) -> str:
        uri = f"agentqa://comparisons/{comparison_id}"
        try:
            identifier = self.paths.component(comparison_id, label="comparison_id")
            report, checksum = self._verified_artifact(
                [self.reports / "comparisons" / f"{identifier}.json"],
                ComparisonReport,
            )
            return _resource_document(
                uri,
                {
                    "artifact_type": "ComparisonReport",
                    "artifact_checksum": checksum,
                    "schema_version": "agentqa.comparison-report/v1alpha1",
                    "artifact": report,
                },
            )
        except Exception as exc:
            return _resource_error(uri, exc)

    def _verified_artifact(self, paths: Sequence[Path], model: type[_Model]) -> tuple[_Model, str]:
        """Load one explicitly addressed, checksummed artifact without generic file access."""
        matches = [path for path in paths if path.is_file() and not path.is_symlink()]
        if len(matches) != 1:
            raise MCPApplicationError(
                "artifact resource was not found or is ambiguous",
                code="MCP_RESOURCE_NOT_FOUND",
            )
        path = matches[0]
        try:
            path.resolve().relative_to(self.paths.root)
            payload = path.read_bytes()
            expected = path.with_name(f"{path.name}.sha256").read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            raise MCPApplicationError(
                "artifact resource could not be verified",
                code="MCP_RESOURCE_CHECKSUM_ERROR",
            ) from None
        if len(payload) > _MAX_REPORT_BYTES or expected != sha256_digest(payload):
            raise MCPApplicationError(
                "artifact resource checksum is invalid",
                code="MCP_RESOURCE_CHECKSUM_ERROR",
            )
        try:
            return model.model_validate_json(payload), expected
        except ValidationError:
            raise MCPApplicationError(
                "artifact resource does not match its shared schema",
                code="MCP_RESOURCE_SCHEMA_ERROR",
            ) from None

    def analysis_resource(self, analysis_id: str) -> str:
        """Expose a Web- or MCP-produced AnalysisReport through one shared schema."""
        safe_id = self.paths.component(analysis_id, label="analysis_id")
        uri = f"agentqa://analyses/{safe_id}"
        try:
            roots = [self.paths.root / ".agentqa" / "artifacts", self.paths.root / "artifacts"]
            candidates = [self.reports / "analyses" / f"{safe_id}.json"]
            for root in roots:
                candidates.extend(root.glob(f"workflows/*/analyses/analysis-{safe_id}.json"))
            report, checksum = self._verified_artifact(candidates, AnalysisReport)
            return _resource_document(
                uri,
                {
                    "artifact_type": "AnalysisReport",
                    "artifact_checksum": checksum,
                    "schema_version": report.schema_version,
                    "artifact": report,
                },
            )
        except Exception as exc:
            return _resource_error(uri, exc)

    def handoff_resource(self, workflow_id: str, handoff_id: str) -> str:
        """Expose a bounded Web handoff to a Coding Agent without filesystem access."""
        safe_workflow = self.paths.component(workflow_id, label="workflow_id")
        safe_handoff = self.paths.component(handoff_id, label="handoff_id")
        uri = f"agentqa://workflows/{safe_workflow}/handoffs/{safe_handoff}"
        try:
            relative = (
                Path("workflows")
                / safe_workflow
                / "handoffs"
                / safe_handoff
                / "handoff-bundle.json"
            )
            candidates = [
                self.paths.root / ".agentqa" / "artifacts" / relative,
                self.paths.root / "artifacts" / relative,
            ]
            handoff, checksum = self._verified_artifact(candidates, HandoffBundle)
            return _resource_document(
                uri,
                {
                    "artifact_type": "HandoffBundle",
                    "artifact_checksum": checksum,
                    "schema_version": handoff.schema_version,
                    "artifact": handoff,
                },
            )
        except Exception as exc:
            return _resource_error(uri, exc)

    @staticmethod
    def _checked_report(path: Path) -> JsonValue:
        checksum = path.with_name(f"{path.name}.sha256")
        try:
            payload = path.read_bytes()
            expected = checksum.read_text(encoding="utf-8").strip()
        except OSError:
            raise MCPApplicationError(
                "report resource was not found",
                code="MCP_RESOURCE_NOT_FOUND",
            ) from None
        if len(payload) > _MAX_REPORT_BYTES:
            raise MCPApplicationError("report resource is too large", code="MCP_RESOURCE_TOO_LARGE")
        if expected != sha256_digest(payload):
            raise MCPApplicationError(
                "report resource checksum mismatch",
                code="MCP_RESOURCE_CHECKSUM_ERROR",
            )
        try:
            return cast(JsonValue, json.loads(payload))
        except (json.JSONDecodeError, UnicodeError):
            raise MCPApplicationError(
                "report resource is not valid JSON",
                code="MCP_RESOURCE_FORMAT_ERROR",
            ) from None

    def validate_manifest(self, manifest_path: str | None = None) -> dict[str, Any]:
        try:
            path = self.paths.resolve(manifest_path or self.manifest_path, file_only=True)
            manifest = _single_spec(path, AgentManifest, "manifest")
        except SpecLoadError as exc:
            return {"valid": False, "diagnostics": _diagnostics(exc, self.paths)}
        return {
            "valid": True,
            "manifest_name": manifest.metadata.name,
            "manifest_hash": hash_spec(manifest),
            "resource_uri": "agentqa://project/manifest",
            "diagnostics": [],
        }

    def validate_tests(self, tests_path: str) -> dict[str, Any]:
        try:
            path = self.paths.resolve(tests_path)
            loaded = load_specs(path)
        except SpecLoadError as exc:
            return {"valid": False, "diagnostics": _diagnostics(exc, self.paths)}
        tests = [item for item in loaded if isinstance(item.spec, AgentTestSpec)]
        if not tests:
            return {
                "valid": False,
                "diagnostics": [
                    {
                        "code": "MCP_TESTS_MISSING",
                        "message": "path does not contain an AgentTest",
                    }
                ],
            }
        return {
            "valid": True,
            "count": len(tests),
            "tests": [{"name": item.spec.metadata.name, "hash": item.spec_hash} for item in tests],
            "diagnostics": [],
        }

    async def generate_tests(
        self,
        *,
        manifest_path: str | None = None,
        goal_path: str | None = None,
        mutations_path: str | None = None,
        code_evidence: Sequence[CodeEvidence] = (),
    ) -> dict[str, Any]:
        manifest = self._manifest(manifest_path)
        goal = (
            GenerationGoal.model_validate({"metadata": {"name": "mcp-generation"}})
            if goal_path is None
            else _single_spec(
                self.paths.resolve(goal_path, file_only=True),
                GenerationGoal,
                "generation goal",
            )
        )
        if code_evidence:
            payload = goal.model_dump(mode="json", by_alias=True)
            payload["codeEvidence"] = [
                *payload.get("codeEvidence", []),
                *(item.model_dump(mode="json", by_alias=True) for item in code_evidence),
            ]
            goal = GenerationGoal.model_validate(payload)
        mutations: list[MutationSpec] = []
        if mutations_path is not None:
            documents = load_specs(self.paths.resolve(mutations_path))
            mutations = [item.spec for item in documents if isinstance(item.spec, MutationSpec)]
        service = GenerationService(self._require_provider())
        plan = await service.generate_plan(manifest, goal)
        report = await service.generate_tests(manifest, plan, mutations)
        root = self.reports / "drafts" / plan.generation_id
        plan_path = write_generation_plan(plan, root / "plan.json", workspace=self.paths.root)
        written = write_generation_suite(report, root / "tests", workspace=self.paths.root)
        return {
            "generation_id": plan.generation_id,
            "test_count": len(report.cases),
            "review_required": sum(
                case.review_status == "REVIEW_REQUIRED" for case in report.cases
            ),
            "coverage_gaps": len(report.gaps),
            "plan_path": self.paths.relative(plan_path),
            "tests_directory": self.paths.relative(written[-1].parent),
            "report_path": self.paths.relative(written[-1]),
        }

    def _authorize_manifest(
        self,
        manifest_path: str | None,
        *,
        allow_side_effects: bool,
        approval_id: str | None,
    ) -> AgentManifest:
        manifest = self._manifest(manifest_path)
        side_effects = [tool for tool in manifest.tools if tool.side_effect != "none"]
        external = [
            tool for tool in side_effects if tool.side_effect in {"external-write", "irreversible"}
        ]
        if side_effects and not allow_side_effects:
            raise MCPApplicationError(
                "run is blocked because Manifest tools declare side effects; set "
                "allowSideEffects=true only after review",
                code="MCP_SIDE_EFFECTS_NOT_ALLOWED",
            )
        if external and (approval_id is None or not approval_id.strip() or len(approval_id) > 256):
            raise MCPApplicationError(
                "external or irreversible tools require a bounded approvalId",
                code="MCP_APPROVAL_REQUIRED",
            )
        return manifest

    async def _run(
        self,
        *,
        tests_path: str,
        target_path: str,
        manifest_path: str | None,
        mutations_path: str | None,
        allow_side_effects: bool,
        approval_id: str | None,
        random_seed: int,
        concurrency: int,
    ) -> dict[str, Any]:
        self._authorize_manifest(
            manifest_path,
            allow_side_effects=allow_side_effects,
            approval_id=approval_id,
        )
        config = RunExecutionConfig(
            tests_path=self.paths.resolve(tests_path),
            target_path=self.paths.resolve(target_path),
            mutations_path=(None if mutations_path is None else self.paths.resolve(mutations_path)),
            manifest_path=self.paths.resolve(
                manifest_path or self.manifest_path,
                file_only=True,
            ),
            artifact_directory=self.artifacts,
            random_seed=random_seed,
            concurrency=concurrency,
            trace_receiver=False,
            judge_provider=self.provider,
            original_command="agentqa mcp run",
        )
        result = await self._run_executor(config)
        return {
            "run_id": result.run_id,
            "status": result.status,
            "exit_code": result.exit_code,
            "summary": _tool_run_summary(result.summary),
            "mutation_score": (
                None
                if result.mutation_score is None
                else result.mutation_score.model_dump(mode="json")
            ),
            "summary_resource_uri": f"agentqa://runs/{result.run_id}/summary",
            "trace_summary_resource_uri": (f"agentqa://runs/{result.run_id}/trace-summary"),
        }

    async def run_tests(
        self,
        *,
        tests_path: str,
        target_path: str,
        manifest_path: str | None = None,
        allow_side_effects: bool = False,
        approval_id: str | None = None,
        random_seed: int = 0,
        concurrency: int = 1,
    ) -> dict[str, Any]:
        return await self._run(
            tests_path=tests_path,
            target_path=target_path,
            manifest_path=manifest_path,
            mutations_path=None,
            allow_side_effects=allow_side_effects,
            approval_id=approval_id,
            random_seed=random_seed,
            concurrency=concurrency,
        )

    async def run_mutations(
        self,
        *,
        tests_path: str,
        target_path: str,
        mutations_path: str,
        manifest_path: str | None = None,
        allow_side_effects: bool = False,
        approval_id: str | None = None,
        random_seed: int = 0,
        concurrency: int = 1,
    ) -> dict[str, Any]:
        return await self._run(
            tests_path=tests_path,
            target_path=target_path,
            manifest_path=manifest_path,
            mutations_path=mutations_path,
            allow_side_effects=allow_side_effects,
            approval_id=approval_id,
            random_seed=random_seed,
            concurrency=concurrency,
        )

    def get_run_result(self, run_id: str) -> dict[str, Any]:
        artifact = self._load_run(run_id)
        failures = [result for result in artifact.results if not result.passed]
        return {
            "run_id": artifact.run_id,
            "summary": _tool_run_summary(artifact.summary),
            "mutation_score": (
                None
                if artifact.mutation_score is None
                else artifact.mutation_score.model_dump(mode="json")
            ),
            "failure_count": len(failures),
            "summary_resource_uri": f"agentqa://runs/{run_id}/summary",
            "trace_summary_resource_uri": f"agentqa://runs/{run_id}/trace-summary",
            "failure_resource_uris": [
                f"agentqa://runs/{run_id}/failures/{result.trial_id}" for result in failures
            ],
        }

    async def analyze_failure(
        self,
        *,
        run_id: str,
        trial_id: str,
        code_evidence: Sequence[CodeEvidence] = (),
    ) -> dict[str, Any]:
        inputs = load_failure_analysis_input(self._existing_run_directory(run_id), trial_id)
        report = await FailureAnalysisService(self._require_provider()).analyze(
            run_id=inputs.run_id,
            test_spec=inputs.test_spec,
            result=inputs.result,
            trace=inputs.trace,
            manifest=inputs.manifest,
            code_evidence=code_evidence,
        )
        destination = write_analysis_report(
            report,
            self.reports / "analyses" / f"{report.analysis_id}.json",
            workspace=self.paths.root,
        )
        return {
            "analysis_id": report.analysis_id,
            "run_id": report.run_id,
            "trial_id": report.trial_id,
            "root_cause_count": len(report.root_causes),
            "missing_evidence_count": len(report.missing_evidence),
            "evidence_hash": report.evidence_hash,
            "report_path": self.paths.relative(destination),
            "failure_resource_uri": f"agentqa://runs/{run_id}/failures/{trial_id}",
        }

    def compare_runs(self, baseline_run_id: str, candidate_run_id: str) -> dict[str, Any]:
        report = compare_runs(
            self._load_run(baseline_run_id),
            self._load_run(candidate_run_id),
        )
        comparison_id = new_id()
        destination = write_report(
            self.reports / "comparisons" / f"{comparison_id}.json",
            report,
        )
        return {
            "comparison_id": comparison_id,
            "baseline_run_id": report.baseline_run_id,
            "candidate_run_id": report.candidate_run_id,
            "new_regressions": report.counts.new_regressions,
            "fixed": report.counts.fixed,
            "quality_cost_status": report.quality_cost.status,
            "report_path": self.paths.relative(destination),
            "resource_uri": f"agentqa://comparisons/{comparison_id}",
        }

    def evaluate_gate(
        self,
        *,
        gate_path: str,
        candidate_run_id: str,
        baseline_run_id: str | None = None,
    ) -> dict[str, Any]:
        gate = _single_spec(
            self.paths.resolve(gate_path, file_only=True),
            QualityGateSpec,
            "gate",
        )
        candidate = self._load_run(candidate_run_id)
        comparison = None
        if baseline_run_id is not None:
            comparison = compare_runs(
                self._load_run(baseline_run_id),
                candidate,
            )
        result = evaluate_gate(gate, candidate, comparison)
        report_id = new_id()
        destination = write_report(self.reports / "gates" / f"{report_id}.json", result)
        return {
            "gate_report_id": report_id,
            "passed": result.passed,
            "exit_code": result.exit_code,
            "rules": [{"name": rule.name, "passed": rule.passed} for rule in result.rules],
            "report_path": self.paths.relative(destination),
        }


__all__ = [
    "MCP_RESOURCE_SCHEMA_VERSION",
    "AgentQAMCPApplication",
    "MCPApplicationError",
    "WorkspacePaths",
]
