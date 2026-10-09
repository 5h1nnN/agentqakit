"""Resource budget evaluation without inventing missing metrics."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from agentqa.evaluators.base import EvaluationRequest
from agentqa.evaluators.utils import TraceView, evaluator_result
from agentqa.models import EvaluationResult, TraceEvent, Violation

_TOTAL_TOKEN_KEYS = {
    "gen_ai.usage.total_tokens",
    "llm.token_count.total",
    "llm.usage.total_tokens",
    "total_tokens",
}
_INPUT_TOKEN_KEYS = {
    "gen_ai.usage.input_tokens",
    "llm.token_count.prompt",
    "llm.usage.prompt_tokens",
    "input_tokens",
    "prompt_tokens",
}
_OUTPUT_TOKEN_KEYS = {
    "gen_ai.usage.output_tokens",
    "llm.token_count.completion",
    "llm.usage.completion_tokens",
    "completion_tokens",
    "output_tokens",
}
_COST_KEYS = {"cost_usd", "gen_ai.usage.cost_usd", "llm.cost_usd"}
_RETRY_KEYS = {"retry_count", "retries", "agentqa.retry.count"}


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if value < 0:
        return None
    return cast(int | float, value)


def _mapping_number(values: Mapping[str, Any], keys: set[str]) -> int | float | None:
    for key, value in values.items():
        if key.lower() in keys:
            number = _number(value)
            if number is not None:
                return number
    return None


def _span_tokens(events: Sequence[TraceEvent]) -> int | None:
    if not events:
        return 0
    total = 0
    for event in events:
        explicit = _mapping_number(event.attributes, _TOTAL_TOKEN_KEYS)
        if explicit is not None:
            total += int(explicit)
            continue
        input_tokens = _mapping_number(event.attributes, _INPUT_TOKEN_KEYS)
        output_tokens = _mapping_number(event.attributes, _OUTPUT_TOKEN_KEYS)
        if input_tokens is not None or output_tokens is not None:
            total += int(input_tokens or 0) + int(output_tokens or 0)
            continue
        return None
    return total


def _span_sum(
    events: Sequence[TraceEvent],
    keys: set[str],
    *,
    require_every: bool = False,
) -> float | None:
    values: list[int | float] = []
    for event in events:
        number = _mapping_number(event.attributes, keys)
        if number is None:
            if require_every:
                return None
            continue
        values.append(number)
    return sum(float(value) for value in values) if values else None


class BudgetEvaluator:
    """Check only metrics that are directly observed in results or traces."""

    type = "budget"
    requires_trace = False
    runs_on_target_error = True

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        budget = request.test_spec.expect.budgets
        if budget is None:
            raise ValueError("budget evaluator requires expect.budgets")
        view = TraceView(request.trace_record)
        target = request.target_result
        latency_ms = (target.ended_at - target.started_at).total_seconds() * 1_000

        total_tokens_value = _mapping_number(target.metadata, {"total_tokens"})
        total_tokens: int | None
        if total_tokens_value is not None:
            total_tokens = int(total_tokens_value)
        elif request.trace_record.status == "complete":
            total_tokens = _span_tokens(view.llm_calls)
        else:
            total_tokens = None
        cost_value = _mapping_number(target.metadata, _COST_KEYS)
        cost_usd: float | None
        if cost_value is not None:
            cost_usd = float(cost_value)
        elif request.trace_record.status == "complete":
            cost_usd = _span_sum(view.llm_calls, _COST_KEYS, require_every=True)
        else:
            cost_usd = None
        retry_value = _mapping_number(target.metadata, _RETRY_KEYS)
        if retry_value is not None:
            retry_count: int | float | None = int(retry_value)
        elif request.trace_record.status == "complete":
            retry_count = _span_sum(view.internal_events, _RETRY_KEYS)
        else:
            retry_count = None
        if retry_count is not None:
            retry_count = int(retry_count)

        steps_value = _mapping_number(target.metadata, {"step_count", "steps"})
        llm_calls_value = _mapping_number(target.metadata, {"llm_calls"})
        tool_calls_value = _mapping_number(target.metadata, {"tool_calls"})
        step_count = (
            int(steps_value)
            if steps_value is not None
            else view.step_count
            if request.trace_record.status == "complete"
            else None
        )
        llm_call_count = (
            int(llm_calls_value)
            if llm_calls_value is not None
            else len(view.llm_calls)
            if request.trace_record.status == "complete"
            else None
        )
        tool_call_count = (
            int(tool_calls_value)
            if tool_calls_value is not None
            else len(view.tool_calls)
            if request.trace_record.status == "complete"
            else None
        )

        violations: list[Violation] = []
        unobservable: list[str] = []
        if budget.max_latency_ms is not None and latency_ms > budget.max_latency_ms:
            violations.append(
                Violation(
                    type="max_latency_ms",
                    message=(f"latency {latency_ms:.3f} ms exceeded {budget.max_latency_ms} ms"),
                    details={"actual": latency_ms, "maximum": budget.max_latency_ms},
                )
            )
        if budget.max_total_tokens is not None:
            if total_tokens is None:
                unobservable.append("total token usage was not recorded")
            elif total_tokens > budget.max_total_tokens:
                violations.append(
                    Violation(
                        type="max_total_tokens",
                        message=(f"token usage {total_tokens} exceeded {budget.max_total_tokens}"),
                        details={"actual": total_tokens, "maximum": budget.max_total_tokens},
                    )
                )

        if budget.max_steps is not None:
            if step_count is None:
                unobservable.append("step count was not recorded")
            elif step_count > budget.max_steps:
                violations.append(
                    Violation(
                        type="max_steps",
                        message=f"step count {step_count} exceeded {budget.max_steps}",
                        details={"actual": step_count, "maximum": budget.max_steps},
                    )
                )
        if budget.max_llm_calls is not None:
            if llm_call_count is None:
                unobservable.append("LLM call count was not recorded")
            elif llm_call_count > budget.max_llm_calls:
                violations.append(
                    Violation(
                        type="max_llm_calls",
                        message=(
                            f"LLM call count {llm_call_count} exceeded {budget.max_llm_calls}"
                        ),
                        details={
                            "actual": llm_call_count,
                            "maximum": budget.max_llm_calls,
                        },
                    )
                )
        if budget.max_tool_calls is not None:
            if tool_call_count is None:
                unobservable.append("tool call count was not recorded")
            elif tool_call_count > budget.max_tool_calls:
                violations.append(
                    Violation(
                        type="max_tool_calls",
                        message=(
                            f"tool call count {tool_call_count} exceeded {budget.max_tool_calls}"
                        ),
                        details={
                            "actual": tool_call_count,
                            "maximum": budget.max_tool_calls,
                        },
                    )
                )

        if budget.max_cost_usd is not None:
            if cost_usd is None:
                unobservable.append("cost in USD was not recorded")
            elif cost_usd > budget.max_cost_usd:
                violations.append(
                    Violation(
                        type="max_cost_usd",
                        message=f"cost {cost_usd:.6f} USD exceeded {budget.max_cost_usd:.6f} USD",
                        details={"actual": cost_usd, "maximum": budget.max_cost_usd},
                    )
                )
        if budget.max_retries is not None:
            if retry_count is None:
                unobservable.append("retry count was not recorded")
            elif retry_count > budget.max_retries:
                violations.append(
                    Violation(
                        type="max_retries",
                        message=f"retry count {retry_count} exceeded {budget.max_retries}",
                        details={"actual": retry_count, "maximum": budget.max_retries},
                    )
                )

        return evaluator_result(
            self.type,
            violations=violations,
            unobservable=list(dict.fromkeys(unobservable)),
            metadata={
                "latency_ms": latency_ms,
                "total_tokens": total_tokens,
                "steps": step_count,
                "llm_calls": llm_call_count,
                "tool_calls": tool_call_count,
                "cost_usd": cost_usd,
                "retries": retry_count,
            },
        )
