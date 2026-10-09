"""Declarative AgentQAKit specification models."""

from typing import Annotated, Literal, TypeAlias

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


class StateExpectation(SpecModel):
    """State assertions retained as a restricted DSL for later evaluation."""

    invariants: list[NonEmptyString] = Field(default_factory=list)


class BudgetExpectation(SpecModel):
    """Positive execution budgets."""

    max_latency_ms: PositiveInt | None = None
    max_total_tokens: PositiveInt | None = None
    max_steps: PositiveInt | None = None
    max_cost_usd: Annotated[float, Field(gt=0)] | None = None


class Expectations(SpecModel):
    """Expected output, trajectory, state, and resource use."""

    output: OutputExpectation | None = None
    tools: ToolExpectation | None = None
    state: StateExpectation | None = None
    budgets: BudgetExpectation | None = None

    @model_validator(mode="after")
    def require_at_least_one_expectation(self) -> "Expectations":
        if all(value is None for value in (self.output, self.tools, self.state, self.budgets)):
            msg = "expect must define output, tools, state, or budgets"
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
