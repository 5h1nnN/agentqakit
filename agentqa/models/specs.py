"""Declarative AgentQAKit specification models."""

import json
from typing import Annotated, Literal, TypeAlias

import regex
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import AnyHttpUrl, Field, JsonValue, StringConstraints, model_validator

from agentqa.models.base import (
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    ResourceName,
    SpecModel,
    UnitInterval,
)

ApiVersion: TypeAlias = Literal["agentqa.dev/v1alpha1"]
ManifestApiVersion: TypeAlias = Literal["agentqa.dev/v1alpha2"]
ManifestText: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
]
ManifestPath: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1_024),
]
Entrypoint: TypeAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$",
    ),
]


def _bounded_json_size(value: JsonValue, *, label: str, max_bytes: int = 65_536) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")


class Metadata(SpecModel):
    """Common metadata for named specification resources."""

    name: ResourceName
    labels: dict[str, str] = Field(default_factory=dict)
    version: NonEmptyString | None = None


class Message(SpecModel):
    """One conversation message supplied to an agent."""

    role: Literal["system", "user", "assistant", "tool"]
    content: JsonValue
    name: NonEmptyString | None = None


class Scenario(SpecModel):
    """Input and initial state for one agent test."""

    input: dict[str, JsonValue] = Field(default_factory=dict)
    initial_state: dict[str, JsonValue] = Field(default_factory=dict)
    messages: list[Message] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_input_or_message(self) -> "Scenario":
        if not self.input and not self.messages:
            msg = "scenario must define at least one input value or message"
            raise ValueError(msg)
        return self


class OutputExpectation(SpecModel):
    """Deterministic assertions over the final output."""

    must_refuse: bool | None = None
    required_patterns: list[NonEmptyString] = Field(default_factory=list)
    forbidden_patterns: list[NonEmptyString] = Field(default_factory=list)
    exact_fields: dict[str, JsonValue] = Field(default_factory=dict)
    json_schema: dict[str, JsonValue] | None = None
    require_non_empty: bool | None = None
    allowed_http_statuses: list[Annotated[int, Field(ge=100, le=599)]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_patterns_and_schema(self) -> "OutputExpectation":
        for pattern in [*self.required_patterns, *self.forbidden_patterns]:
            if len(pattern) > 1_024:
                msg = "output patterns must not exceed 1024 characters"
                raise ValueError(msg)
            try:
                regex.compile(pattern)
            except regex.error:
                msg = "output pattern is not a valid regular expression"
                raise ValueError(msg) from None
        if self.json_schema is not None:
            try:
                Draft202012Validator.check_schema(self.json_schema)
            except SchemaError:
                msg = "jsonSchema is not a valid Draft 2020-12 schema"
                raise ValueError(msg) from None
        if len(self.allowed_http_statuses) != len(set(self.allowed_http_statuses)):
            msg = "allowedHttpStatuses must not contain duplicates"
            raise ValueError(msg)
        return self


class OrderConstraint(SpecModel):
    """A partial ordering requirement between two tool calls."""

    before: ResourceName
    after: ResourceName


class ToolExpectation(SpecModel):
    """Allowed tool-call behavior for a test."""

    required: list[ResourceName] = Field(default_factory=list)
    forbidden: list[ResourceName] = Field(default_factory=list)
    min_calls: PositiveInt | dict[str, NonNegativeInt] | None = None
    max_calls: NonNegativeInt | dict[str, NonNegativeInt] | None = None
    order: list[OrderConstraint] = Field(default_factory=list)
    argument_schemas: dict[ResourceName, dict[str, JsonValue]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_argument_schemas(self) -> "ToolExpectation":
        if not (
            self.required
            or self.forbidden
            or self.min_calls is not None
            or self.max_calls is not None
            or self.order
            or self.argument_schemas
        ):
            msg = "tools must define at least one constraint"
            raise ValueError(msg)
        for schema in self.argument_schemas.values():
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError:
                msg = "tool argumentSchemas values must be valid Draft 2020-12 schemas"
                raise ValueError(msg) from None
        return self


TraceKind: TypeAlias = Literal["AGENT", "LLM", "TOOL", "RETRIEVER", "NODE", "EVALUATOR"]


class SpanAttributeExpectation(SpecModel):
    """An attribute required on every span selected by optional kind and name."""

    attribute: NonEmptyString
    kind: TraceKind | None = None
    name: NonEmptyString | None = None
    equals: JsonValue = None


class TrajectoryExpectation(SpecModel):
    """Assertions over normalized trace shape and span metadata."""

    max_error_spans: NonNegativeInt | None = None
    required_span_attributes: list[SpanAttributeExpectation] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_constraint(self) -> "TrajectoryExpectation":
        if self.max_error_spans is None and not self.required_span_attributes:
            msg = "trajectory must define maxErrorSpans or requiredSpanAttributes"
            raise ValueError(msg)
        return self


StateOperator: TypeAlias = Literal["equals", "notEquals", "exists", "in", "before"]


class StateInvariant(SpecModel):
    """One non-executable state assertion over a restricted field path."""

    field: NonEmptyString
    op: StateOperator
    value: JsonValue = None

    @model_validator(mode="after")
    def validate_operator_value(self) -> "StateInvariant":
        has_value = "value" in self.model_fields_set
        if self.op == "exists" and has_value:
            msg = "exists state invariant does not accept value"
            raise ValueError(msg)
        if self.op != "exists" and not has_value:
            msg = f"{self.op} state invariant requires value"
            raise ValueError(msg)
        if self.op == "in" and not isinstance(self.value, list):
            msg = "in state invariant requires a list value"
            raise ValueError(msg)
        if self.op == "before" and not isinstance(self.value, str):
            msg = "before state invariant requires another field path as value"
            raise ValueError(msg)
        return self


class StateExpectation(SpecModel):
    """A conjunction of non-executable state assertions."""

    all: list[StateInvariant] = Field(min_length=1)


class BudgetExpectation(SpecModel):
    """Positive execution budgets."""

    max_latency_ms: PositiveInt | None = None
    max_total_tokens: PositiveInt | None = None
    max_steps: PositiveInt | None = None
    max_cost_usd: Annotated[float, Field(gt=0)] | None = None
    max_llm_calls: NonNegativeInt | None = None
    max_tool_calls: NonNegativeInt | None = None
    max_retries: NonNegativeInt | None = None

    @model_validator(mode="after")
    def require_constraint(self) -> "BudgetExpectation":
        if all(
            value is None
            for value in (
                self.max_latency_ms,
                self.max_total_tokens,
                self.max_steps,
                self.max_cost_usd,
                self.max_llm_calls,
                self.max_tool_calls,
                self.max_retries,
            )
        ):
            msg = "budgets must define at least one limit"
            raise ValueError(msg)
        return self


class Expectations(SpecModel):
    """Expected output, trajectory, state, and resource use."""

    output: OutputExpectation | None = None
    tools: ToolExpectation | None = None
    trajectory: TrajectoryExpectation | None = None
    state: StateExpectation | None = None
    budgets: BudgetExpectation | None = None

    @model_validator(mode="after")
    def require_at_least_one_expectation(self) -> "Expectations":
        if all(
            value is None
            for value in (
                self.output,
                self.tools,
                self.trajectory,
                self.state,
                self.budgets,
            )
        ):
            msg = "expect must define output, tools, trajectory, state, or budgets"
            raise ValueError(msg)
        return self


EvaluatorType: TypeAlias = Literal[
    "schema",
    "text",
    "tool_call",
    "trajectory",
    "state",
    "budget",
    "privacy",
    "llm_judge",
]


class EvaluatorSpec(SpecModel):
    """Configuration for one known evaluator implementation."""

    type: EvaluatorType
    rubric: NonEmptyString | None = None
    config: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_judge_rubric(self) -> "EvaluatorSpec":
        if self.type == "llm_judge" and self.rubric is None:
            msg = "llm_judge evaluator requires rubric"
            raise ValueError(msg)
        if self.type != "llm_judge" and self.rubric is not None:
            msg = "rubric is only valid for llm_judge"
            raise ValueError(msg)
        return self


class TrialPolicy(SpecModel):
    """How multiple trials determine one test-case outcome."""

    strategy: Literal["all", "majority", "minPassRate"] = "all"
    min_pass_rate: UnitInterval | None = None

    @model_validator(mode="after")
    def validate_threshold(self) -> "TrialPolicy":
        if self.strategy == "minPassRate" and self.min_pass_rate is None:
            msg = "minPassRate strategy requires minPassRate"
            raise ValueError(msg)
        if self.strategy != "minPassRate" and self.min_pass_rate is not None:
            msg = "minPassRate is only valid with the minPassRate strategy"
            raise ValueError(msg)
        return self


class AgentTestSpec(SpecModel):
    """A single declarative agent test case."""

    api_version: ApiVersion = Field(default="agentqa.dev/v1alpha1", alias="apiVersion")
    kind: Literal["AgentTest"] = "AgentTest"
    metadata: Metadata
    target_ref: ResourceName | None = None
    scenario: Scenario
    mutation: ResourceName | None = None
    expect: Expectations
    evaluators: list[EvaluatorSpec] = Field(min_length=1)
    trials: PositiveInt = 1
    trial_policy: TrialPolicy = Field(default_factory=TrialPolicy)

    @model_validator(mode="after")
    def protect_high_risk_judgements(self) -> "AgentTestSpec":
        risk = self.metadata.labels.get("risk", "").strip().lower()
        if risk in {"high", "critical"} and all(
            evaluator.type == "llm_judge" for evaluator in self.evaluators
        ):
            msg = "high-risk tests cannot rely only on llm_judge"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_evaluator_inputs(self) -> "AgentTestSpec":
        evaluator_types = [evaluator.type for evaluator in self.evaluators]
        if len(evaluator_types) != len(set(evaluator_types)):
            msg = "evaluator types must be unique within one AgentTest"
            raise ValueError(msg)

        output = self.expect.output
        for evaluator in self.evaluators:
            if evaluator.type == "schema" and (
                output is None
                or not (
                    output.exact_fields
                    or output.json_schema is not None
                    or output.require_non_empty is not None
                    or output.allowed_http_statuses
                )
            ):
                msg = "schema evaluator requires a structural output expectation"
                raise ValueError(msg)
            if evaluator.type == "text" and (
                output is None
                or not (
                    output.required_patterns
                    or output.forbidden_patterns
                    or output.must_refuse is not None
                )
            ):
                msg = "text evaluator requires a text or refusal output expectation"
                raise ValueError(msg)
            if evaluator.type == "tool_call" and self.expect.tools is None:
                msg = "tool_call evaluator requires expect.tools"
                raise ValueError(msg)
            if evaluator.type == "trajectory" and self.expect.trajectory is None:
                msg = "trajectory evaluator requires expect.trajectory"
                raise ValueError(msg)
            if evaluator.type == "state" and self.expect.state is None:
                msg = "state evaluator requires expect.state"
                raise ValueError(msg)
            if evaluator.type == "budget" and self.expect.budgets is None:
                msg = "budget evaluator requires expect.budgets"
                raise ValueError(msg)
            if evaluator.type == "privacy":
                configured = evaluator.config.get("forbiddenPatterns")
                if configured is not None and (
                    not isinstance(configured, list)
                    or not all(isinstance(item, str) and item for item in configured)
                ):
                    msg = "privacy forbiddenPatterns must be a list of non-empty strings"
                    raise ValueError(msg)
                has_patterns = bool(
                    (output is not None and output.forbidden_patterns) or configured
                )
                if not has_patterns and (output is None or output.must_refuse is None):
                    msg = "privacy evaluator requires forbidden patterns or refusal intent"
                    raise ValueError(msg)
        return self


class HttpRequestSpec(SpecModel):
    """HTTP request template."""

    headers: dict[str, str] = Field(default_factory=dict)
    query: dict[str, str] = Field(default_factory=dict)
    json_body: JsonValue = Field(default=None, alias="json")


class HttpResponseSpec(SpecModel):
    """HTTP response extraction rules."""

    output_path: NonEmptyString = "$.answer"
    allowed_headers: list[NonEmptyString] = Field(default_factory=list)


class TargetSpecBase(SpecModel):
    """Fields shared by all target adapters."""

    api_version: ApiVersion = Field(default="agentqa.dev/v1alpha1", alias="apiVersion")
    kind: Literal["Target"] = "Target"
    metadata: Metadata
    timeout_ms: PositiveInt = 30_000


class HttpTargetSpec(TargetSpecBase):
    """A generic HTTP target."""

    type: Literal["http"]
    url: AnyHttpUrl
    method: Literal["GET", "POST"] = "POST"
    request: HttpRequestSpec = Field(default_factory=HttpRequestSpec)
    response: HttpResponseSpec = Field(default_factory=HttpResponseSpec)
    allow_remote: bool = False
    allow_private_network: bool = False
    allowed_hosts: list[NonEmptyString] = Field(default_factory=list)
    connect_timeout_ms: PositiveInt = 5_000
    read_timeout_ms: PositiveInt = 30_000
    max_response_bytes: PositiveInt = 1_048_576

    @model_validator(mode="after")
    def validate_network_policy(self) -> "HttpTargetSpec":
        if self.allow_private_network and not self.allow_remote:
            msg = "allowPrivateNetwork requires allowRemote"
            raise ValueError(msg)
        return self


class PythonTargetSpec(TargetSpecBase):
    """A Python callable target executed in an isolated worker."""

    type: Literal["python"]
    entrypoint: Entrypoint
    isolation: Literal["subprocess"] = "subprocess"
    worker_mode: Literal["perInvocation", "persistent"] = "perInvocation"
    environment_allowlist: list[NonEmptyString] = Field(default_factory=list)
    working_directory: NonEmptyString | None = None


class LangGraphTargetSpec(TargetSpecBase):
    """A LangGraph object or factory target."""

    type: Literal["langgraph"]
    entrypoint: Entrypoint
    input_key: NonEmptyString = "messages"
    output_path: NonEmptyString | None = None
    state_diff_allowlist: list[NonEmptyString] = Field(default_factory=list)


TargetSpec: TypeAlias = Annotated[
    HttpTargetSpec | PythonTargetSpec | LangGraphTargetSpec,
    Field(discriminator="type"),
]


class MutationTarget(SpecModel):
    """The named interception point selected by a mutation."""

    kind: Literal["tool", "retriever", "prompt", "state", "http"]
    name: ResourceName


class MutationWhen(SpecModel):
    """Optional deterministic mutation matching conditions."""

    call_index: PositiveInt | None = None
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    argument_exists: list[NonEmptyString] = Field(default_factory=list)
    test_labels: dict[str, str] = Field(default_factory=dict)
    probability: UnitInterval = 1.0
    mode: Literal["first", "every"] = "first"


MutationOperationType: TypeAlias = Literal[
    "timeout",
    "tool_timeout",
    "tool_exception",
    "invalid_json",
    "missing_field",
    "wrong_value",
    "duplicate_call_result",
    "stale_result",
    "empty_retrieval",
    "irrelevant_document",
    "indirect_prompt_injection",
    "http_status",
    "http_disconnect",
]


class MutationOperation(SpecModel):
    """A mutation operator and its common parameters."""

    type: MutationOperationType
    delay_ms: PositiveInt | None = None
    message: NonEmptyString | None = None
    field: NonEmptyString | None = None
    value: JsonValue = None
    status_code: Annotated[int, Field(ge=500, le=599)] | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_timeout_delay(self) -> "MutationOperation":
        if self.type in {"timeout", "tool_timeout"} and self.delay_ms is None:
            msg = "timeout mutation requires delayMs"
            raise ValueError(msg)
        if self.type in {"missing_field", "wrong_value"} and self.field is None:
            msg = f"{self.type} mutation requires field"
            raise ValueError(msg)
        if self.type == "wrong_value" and "value" not in self.model_fields_set:
            msg = "wrong_value mutation requires value"
            raise ValueError(msg)
        if self.type == "http_status" and self.status_code is None:
            msg = "http_status mutation requires statusCode"
            raise ValueError(msg)
        if self.type != "http_status" and self.status_code is not None:
            msg = "statusCode is only valid for http_status"
            raise ValueError(msg)
        return self


class ExpectedRecovery(SpecModel):
    """Expected behavior after a mutation is applied."""

    max_retries: NonNegativeInt | None = None
    must_not_claim_success: bool | None = None
    must_explain_temporary_failure: bool | None = None


class MutationSpec(SpecModel):
    """A deterministic fault mutation specification."""

    api_version: ApiVersion = Field(default="agentqa.dev/v1alpha1", alias="apiVersion")
    kind: Literal["Mutation"] = "Mutation"
    metadata: Metadata
    target: MutationTarget
    when: MutationWhen = Field(default_factory=MutationWhen)
    operation: MutationOperation
    expected_recovery: ExpectedRecovery | None = None

    @model_validator(mode="after")
    def validate_target_operation(self) -> "MutationSpec":
        supported: dict[str, set[str]] = {
            "tool": {
                "timeout",
                "tool_timeout",
                "tool_exception",
                "invalid_json",
                "missing_field",
                "wrong_value",
                "duplicate_call_result",
                "stale_result",
            },
            "retriever": {
                "empty_retrieval",
                "irrelevant_document",
                "indirect_prompt_injection",
            },
            "http": {
                "timeout",
                "invalid_json",
                "missing_field",
                "wrong_value",
                "http_status",
                "http_disconnect",
            },
        }
        allowed = supported.get(self.target.kind)
        if allowed is None or self.operation.type not in allowed:
            msg = (
                f"operation {self.operation.type!r} is not supported for "
                f"target kind {self.target.kind!r}"
            )
            raise ValueError(msg)
        return self


class SourceRevision(SpecModel):
    """Source repository identity for an agent build."""

    repository: AnyHttpUrl
    commit: ManifestText
    dirty: bool | None = None
    source_hash: ManifestText | None = None


class RuntimeIdentity(SpecModel):
    """Agent implementation language and framework."""

    language: ManifestText
    framework: ManifestText | None = None
    entrypoint: Entrypoint | None = None
    dependency_hash: ManifestText | None = None


class ModelIdentity(SpecModel):
    """Model provider and generation parameters."""

    provider: ManifestText
    name: ManifestText
    parameters: dict[str, JsonValue] = Field(default_factory=dict, max_length=100)

    @model_validator(mode="after")
    def bound_parameters(self) -> "ModelIdentity":
        _bounded_json_size(self.parameters, label="model parameters")
        return self


class PromptIdentity(SpecModel):
    """Hashes that identify the prompt assets under test."""

    system_prompt_hash: ManifestText | None = None
    asset_hashes: dict[ResourceName, ManifestText] = Field(
        default_factory=dict,
        max_length=200,
    )


class ToolIdentity(SpecModel):
    """A tool identity plus the side-effect semantics needed by test generation."""

    name: ResourceName
    description: ManifestText | None = None
    schema_hash: ManifestText | None = None
    input_schema: dict[str, JsonValue] | None = None
    output_schema: dict[str, JsonValue] | None = None
    side_effect: Literal["none", "local-write", "external-write", "irreversible"] = "none"
    risk: Literal["low", "medium", "high", "critical"] | None = None
    idempotent: bool | None = None
    requires_confirmation: bool = False

    @model_validator(mode="after")
    def validate_tool_semantics(self) -> "ToolIdentity":
        for label, schema in (
            ("inputSchema", self.input_schema),
            ("outputSchema", self.output_schema),
        ):
            if schema is None:
                continue
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError:
                msg = f"tool {label} must be a valid Draft 2020-12 schema"
                raise ValueError(msg) from None
            _bounded_json_size(schema, label=f"tool {label}")
        if self.side_effect != "none" and (self.risk is None or self.idempotent is None):
            msg = "tools with side effects require risk and idempotent"
            raise ValueError(msg)
        return self


class KnowledgeIdentity(SpecModel):
    """Optional knowledge-index identity."""

    enabled: bool
    index_hash: ManifestText | None = None

    @model_validator(mode="after")
    def require_enabled_index_hash(self) -> "KnowledgeIdentity":
        if self.enabled and self.index_hash is None:
            msg = "enabled knowledge requires indexHash"
            raise ValueError(msg)
        return self


class AgentPurpose(SpecModel):
    """Declared product purpose and explicit out-of-scope behavior."""

    summary: ManifestText
    out_of_scope: list[ManifestText] = Field(default_factory=list, max_length=100)


class AgentInterfaces(SpecModel):
    """Public input and output contracts exposed by an agent."""

    input_schema: dict[str, JsonValue] | None = None
    output_schema: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def validate_schemas(self) -> "AgentInterfaces":
        for label, schema in (
            ("inputSchema", self.input_schema),
            ("outputSchema", self.output_schema),
        ):
            if schema is None:
                continue
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError:
                msg = f"interfaces.{label} must be a valid Draft 2020-12 schema"
                raise ValueError(msg) from None
            _bounded_json_size(schema, label=f"interfaces.{label}")
        if self.input_schema is None and self.output_schema is None:
            msg = "interfaces must define inputSchema or outputSchema"
            raise ValueError(msg)
        return self


class AgentCapability(SpecModel):
    """One named capability advertised by the agent."""

    name: ResourceName
    description: ManifestText


class AgentInvariant(SpecModel):
    """A stable business or safety rule used to derive tests."""

    id: ResourceName
    description: ManifestText
    severity: Literal["low", "medium", "high", "critical"]


class AgentPolicies(SpecModel):
    """Business invariants and prohibited or permission-sensitive behavior."""

    invariants: list[AgentInvariant] = Field(default_factory=list, max_length=200)
    forbidden_behaviors: list[ManifestText] = Field(default_factory=list, max_length=100)
    permission_requirements: list[ManifestText] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def require_policy(self) -> "AgentPolicies":
        if not (self.invariants or self.forbidden_behaviors or self.permission_requirements):
            msg = "policies must define at least one constraint"
            raise ValueError(msg)
        ids = [item.id for item in self.invariants]
        if len(ids) != len(set(ids)):
            msg = "policy invariant ids must be unique"
            raise ValueError(msg)
        return self


class HumanInteractionPolicy(SpecModel):
    """Human-in-the-loop behavior and intervention budget."""

    allowed: bool
    required_when: list[ManifestText] = Field(default_factory=list, max_length=100)
    max_interventions: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_allowed_interactions(self) -> "HumanInteractionPolicy":
        if not self.allowed and (self.required_when or self.max_interventions not in {None, 0}):
            msg = "disabled human interaction cannot define requiredWhen or a positive budget"
            raise ValueError(msg)
        return self


class AgentQuality(SpecModel):
    """Quality objectives and default resource budgets for generated tests."""

    objectives: list[ResourceName] = Field(min_length=1, max_length=50)
    budgets: BudgetExpectation | None = None

    @model_validator(mode="after")
    def unique_objectives(self) -> "AgentQuality":
        if len(self.objectives) != len(set(self.objectives)):
            msg = "quality objectives must be unique"
            raise ValueError(msg)
        return self


class AgentTestHints(SpecModel):
    """Non-authoritative scenarios suggested for AI-assisted generation."""

    important_scenarios: list[ManifestText] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def require_hint(self) -> "AgentTestHints":
        if not self.important_scenarios:
            msg = "testHints must define at least one important scenario"
            raise ValueError(msg)
        return self


class GeneratedBy(SpecModel):
    """Optional model identity for inferred manifest content."""

    provider: ResourceName
    model: ManifestText
    prompt_hash: ManifestText | None = None


class ManifestProvenance(SpecModel):
    """Origin and review state for semantic manifest information."""

    default_level: Literal["declared", "discovered", "inferred", "verified"] = "declared"
    declared_by: ManifestText | None = None
    discovered_from: list[ManifestPath] = Field(default_factory=list, max_length=200)
    generated_by: GeneratedBy | None = None
    reviewed: bool = False

    @model_validator(mode="after")
    def validate_origin(self) -> "ManifestProvenance":
        if self.default_level == "inferred" and self.generated_by is None:
            msg = "inferred provenance requires generatedBy"
            raise ValueError(msg)
        if self.default_level == "verified" and not self.reviewed:
            msg = "verified provenance requires reviewed=true"
            raise ValueError(msg)
        return self


class AgentManifest(SpecModel):
    """A reproducible identity and semantic contract for the agent under test."""

    api_version: ManifestApiVersion = Field(default="agentqa.dev/v1alpha2", alias="apiVersion")
    kind: Literal["AgentManifest"] = "AgentManifest"
    metadata: Metadata
    source: SourceRevision | None = None
    runtime: RuntimeIdentity
    model: ModelIdentity | None = None
    prompts: PromptIdentity | None = None
    purpose: AgentPurpose | None = None
    interfaces: AgentInterfaces | None = None
    capabilities: list[AgentCapability] = Field(default_factory=list, max_length=200)
    tools: list[ToolIdentity] = Field(default_factory=list, max_length=500)
    policies: AgentPolicies | None = None
    human_interaction: HumanInteractionPolicy | None = None
    quality: AgentQuality | None = None
    knowledge: KnowledgeIdentity | None = None
    test_hints: AgentTestHints | None = None
    provenance: ManifestProvenance = Field(default_factory=ManifestProvenance)

    @model_validator(mode="after")
    def validate_semantic_contract(self) -> "AgentManifest":
        capability_names = [item.name for item in self.capabilities]
        if len(capability_names) != len(set(capability_names)):
            msg = "capability names must be unique"
            raise ValueError(msg)
        tool_names = [item.name for item in self.tools]
        if len(tool_names) != len(set(tool_names)):
            msg = "tool names must be unique"
            raise ValueError(msg)
        if any(tool.requires_confirmation for tool in self.tools) and (
            self.human_interaction is None or not self.human_interaction.allowed
        ):
            msg = "tools requiring confirmation need humanInteraction.allowed=true"
            raise ValueError(msg)
        _bounded_json_size(
            self.model_dump(mode="json", by_alias=True),
            label="AgentManifest",
            max_bytes=1024 * 1024,
        )
        return self


class QualityGateRules(SpecModel):
    """Optional quality thresholds evaluated against a candidate run."""

    min_pass_rate: UnitInterval | None = None
    min_mutation_score: UnitInterval | None = None
    max_new_regressions: NonNegativeInt | None = None
    max_high_risk_failures: NonNegativeInt | None = None
    max_p95_latency_increase: Annotated[float, Field(ge=0)] | None = None
    max_average_cost_increase: Annotated[float, Field(ge=0)] | None = None
    require_replay_bundle_for_failures: bool | None = None

    @model_validator(mode="after")
    def require_rule(self) -> "QualityGateRules":
        if all(
            value is None
            for value in (
                self.min_pass_rate,
                self.min_mutation_score,
                self.max_new_regressions,
                self.max_high_risk_failures,
                self.max_p95_latency_increase,
                self.max_average_cost_increase,
                self.require_replay_bundle_for_failures,
            )
        ):
            msg = "quality gate rules must define at least one threshold"
            raise ValueError(msg)
        return self


class QualityGateSpec(SpecModel):
    """Declarative quality gate evaluated by ``agentqa gate evaluate``."""

    api_version: ApiVersion = Field(default="agentqa.dev/v1alpha1", alias="apiVersion")
    kind: Literal["QualityGate"] = "QualityGate"
    metadata: Metadata
    rules: QualityGateRules
