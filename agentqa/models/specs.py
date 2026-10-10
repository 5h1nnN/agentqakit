"""Declarative AgentQAKit specification models."""

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
Entrypoint: TypeAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$",
    ),
]


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
    commit: NonEmptyString


class RuntimeIdentity(SpecModel):
    """Agent implementation language and framework."""

    language: NonEmptyString
    framework: NonEmptyString | None = None


class ModelIdentity(SpecModel):
    """Model provider and generation parameters."""

    provider: NonEmptyString
    name: NonEmptyString
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class PromptIdentity(SpecModel):
    """Hashes that identify the prompt assets under test."""

    system_prompt_hash: NonEmptyString | None = None


class ToolIdentity(SpecModel):
    """A tool and its input-schema identity."""

    name: ResourceName
    schema_hash: NonEmptyString | None = None


class KnowledgeIdentity(SpecModel):
    """Optional knowledge-index identity."""

    enabled: bool
    index_hash: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_enabled_index_hash(self) -> "KnowledgeIdentity":
        if self.enabled and self.index_hash is None:
            msg = "enabled knowledge requires indexHash"
            raise ValueError(msg)
        return self


class AgentManifest(SpecModel):
    """A reproducible identity for the agent version under test."""

    api_version: ApiVersion = Field(default="agentqa.dev/v1alpha1", alias="apiVersion")
    kind: Literal["AgentManifest"] = "AgentManifest"
    metadata: Metadata
    source: SourceRevision | None = None
    runtime: RuntimeIdentity
    model: ModelIdentity | None = None
    prompts: PromptIdentity | None = None
    tools: list[ToolIdentity] = Field(default_factory=list)
    knowledge: KnowledgeIdentity | None = None


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
