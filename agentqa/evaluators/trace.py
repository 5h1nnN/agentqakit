"""Tool-call and trajectory evaluators over normalized trace evidence."""

from collections import Counter

from jsonschema import Draft202012Validator

from agentqa.evaluators.base import EvaluationRequest
from agentqa.evaluators.utils import TraceView, evaluator_result, is_content_summary
from agentqa.models import EvaluationResult, Violation


class ToolCallEvaluator:
    """Check tool presence, counts, arguments, and partial ordering."""

    type = "tool_call"
    requires_trace = True
    runs_on_target_error = True

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        expectation = request.test_spec.expect.tools
        if expectation is None:
            raise ValueError("tool_call evaluator requires expect.tools")
        view = TraceView(request.trace_record)
        calls = view.tool_calls
        counts = Counter(call.name for call in calls)
        violations: list[Violation] = []

        for name in expectation.required:
            if counts[name] == 0:
                violations.append(
                    Violation(
                        type="required_tool",
                        message=f"required tool {name!r} was not called",
                        details={"tool": name},
                    )
                )
        for name in expectation.forbidden:
            for call in calls:
                if call.name == name:
                    violations.append(
                        Violation(
                            type="forbidden_tool",
                            message=f"forbidden tool {name!r} was called",
                            span_id=call.event.span_id,
                            details={"tool": name},
                        )
                    )

        if isinstance(expectation.min_calls, int):
            if len(calls) < expectation.min_calls:
                violations.append(
                    Violation(
                        type="min_tool_calls",
                        message=(
                            f"observed {len(calls)} tool call(s), fewer than "
                            f"{expectation.min_calls}"
                        ),
                    )
                )
        elif isinstance(expectation.min_calls, dict):
            for name, minimum in expectation.min_calls.items():
                if counts[name] < minimum:
                    violations.append(
                        Violation(
                            type="min_tool_calls",
                            message=f"tool {name!r} was called fewer than {minimum} time(s)",
                            details={"tool": name, "actual": counts[name], "minimum": minimum},
                        )
                    )
        if isinstance(expectation.max_calls, int):
            if len(calls) > expectation.max_calls:
                violating = calls[expectation.max_calls]
                violations.append(
                    Violation(
                        type="max_tool_calls",
                        message=(
                            f"observed {len(calls)} tool call(s), more than {expectation.max_calls}"
                        ),
                        span_id=violating.event.span_id,
                    )
                )
        elif isinstance(expectation.max_calls, dict):
            for name, maximum in expectation.max_calls.items():
                matching = [call for call in calls if call.name == name]
                if len(matching) > maximum:
                    violations.append(
                        Violation(
                            type="max_tool_calls",
                            message=f"tool {name!r} was called more than {maximum} time(s)",
                            span_id=matching[maximum].event.span_id,
                            details={"tool": name, "actual": len(matching), "maximum": maximum},
                        )
                    )

        positions: dict[str, int] = {}
        for index, call in enumerate(calls):
            positions.setdefault(call.name, index)
        for constraint in expectation.order:
            before = positions.get(constraint.before)
            after = positions.get(constraint.after)
            if before is None or after is None:
                violations.append(
                    Violation(
                        type="tool_order_missing",
                        message=(
                            f"cannot satisfy {constraint.before!r} before "
                            f"{constraint.after!r} because a tool was not called"
                        ),
                    )
                )
            elif before >= after:
                violations.append(
                    Violation(
                        type="tool_order",
                        message=(
                            f"tool {constraint.before!r} did not occur before {constraint.after!r}"
                        ),
                        span_id=calls[after].event.span_id,
                    )
                )

        unobservable: list[str] = []
        for name, schema in expectation.argument_schemas.items():
            matching = [call for call in calls if call.name == name]
            if not matching:
                violations.append(
                    Violation(
                        type="tool_argument_schema_missing_call",
                        message=f"tool {name!r} had no call whose arguments could be checked",
                        details={"tool": name},
                    )
                )
                continue
            validator = Draft202012Validator(schema)
            for call in matching:
                if is_content_summary(call.arguments):
                    unobservable.append(
                        f"arguments for tool {name!r} were retained only as a summary"
                    )
                    continue
                errors = sorted(
                    validator.iter_errors(call.arguments),
                    key=lambda item: tuple(str(part) for part in item.path),
                )
                for error in errors:
                    path = "$" + "".join(f"[{part}]" for part in error.absolute_path)
                    violations.append(
                        Violation(
                            type="tool_argument_schema",
                            message=f"tool {name!r} arguments failed JSON Schema at {path}",
                            span_id=call.event.span_id,
                            details={"tool": name, "path": path},
                        )
                    )

        return evaluator_result(
            self.type,
            violations=violations,
            unobservable=list(dict.fromkeys(unobservable)),
            metadata={
                "tool_call_count": len(calls),
                "tool_call_counts": dict(sorted(counts.items())),
            },
        )


class TrajectoryEvaluator:
    """Check error spans and required span attributes."""

    type = "trajectory"
    requires_trace = True
    runs_on_target_error = True

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        expectation = request.test_spec.expect.trajectory
        if expectation is None:
            raise ValueError("trajectory evaluator requires expect.trajectory")
        view = TraceView(request.trace_record)
        violations: list[Violation] = []
        error_spans = [event for event in view.internal_events if event.status == "ERROR"]
        if (
            expectation.max_error_spans is not None
            and len(error_spans) > expectation.max_error_spans
        ):
            first_extra = error_spans[expectation.max_error_spans]
            violations.append(
                Violation(
                    type="max_error_spans",
                    message=(
                        f"observed {len(error_spans)} ERROR span(s), more than "
                        f"{expectation.max_error_spans}"
                    ),
                    span_id=first_extra.span_id,
                )
            )

        for required in expectation.required_span_attributes:
            selected = [
                event
                for event in view.internal_events
                if (required.kind is None or event.kind == required.kind)
                and (required.name is None or event.name == required.name)
            ]
            if not selected:
                violations.append(
                    Violation(
                        type="required_span_missing",
                        message="no span matched a required attribute selector",
                        details={
                            "attribute": required.attribute,
                            "kind": required.kind,
                            "name": required.name,
                        },
                    )
                )
                continue
            compare_value = "equals" in required.model_fields_set
            for event in selected:
                if required.attribute not in event.attributes:
                    violations.append(
                        Violation(
                            type="required_span_attribute",
                            message=(
                                f"span {event.name!r} did not contain required attribute "
                                f"{required.attribute!r}"
                            ),
                            span_id=event.span_id,
                            details={"attribute": required.attribute},
                        )
                    )
                elif compare_value and event.attributes[required.attribute] != required.equals:
                    violations.append(
                        Violation(
                            type="span_attribute_value",
                            message=(
                                f"span {event.name!r} attribute {required.attribute!r} "
                                "did not equal the expected value"
                            ),
                            span_id=event.span_id,
                            details={"attribute": required.attribute},
                        )
                    )

        return evaluator_result(
            self.type,
            violations=violations,
            metadata={
                "error_span_count": len(error_spans),
                "span_count": len(view.internal_events),
            },
        )
