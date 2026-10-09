"""LangGraph target adapter with optional event-stream observation."""

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Literal

from opentelemetry import trace

from agentqa.conventions import new_id, utc_now
from agentqa.models import (
    ErrorInfo,
    LangGraphTargetSpec,
    RunContext,
    TargetResult,
    TestInput,
    TraceEvent,
)
from agentqa.targets.errors import TargetConfigurationError, TargetErrorCode
from agentqa.targets.utils import extract_path, load_entrypoint, safe_jsonable, to_jsonable
from agentqa.tracing.context import (
    get_current_trace_options,
    get_current_trace_store,
    record_trace_event,
)
from agentqa.tracing.normalization import content_value


@dataclass(slots=True)
class AgentQARuntime:
    """Factory-time extension point for future mutation-aware wrappers."""

    capabilities: dict[str, bool] = field(
        default_factory=lambda: {
            "tool_wrapping": True,
            "retriever_wrapping": True,
            "mutation_engine": False,
        }
    )

    def wrap_tool(self, tool: Any) -> Any:
        """Mark the stable wrapping boundary; T5 supplies mutation behavior."""
        return tool

    def wrap_retriever(self, retriever: Any) -> Any:
        """Mark the stable wrapping boundary; T5 supplies mutation behavior."""
        return retriever


@dataclass(slots=True)
class _OpenSpan:
    span_id: str
    parent_span_id: str | None
    kind: Literal["AGENT", "LLM", "TOOL", "RETRIEVER", "NODE", "EVALUATOR"]
    name: str
    input: Any
    started_at: Any
    attributes: dict[str, Any]


def _event_kind(
    event_name: str, *, is_root: bool
) -> Literal["AGENT", "LLM", "TOOL", "RETRIEVER", "NODE", "EVALUATOR"]:
    if event_name.startswith(("on_chat_model_", "on_llm_")):
        return "LLM"
    if event_name.startswith("on_tool_"):
        return "TOOL"
    if event_name.startswith("on_retriever_"):
        return "RETRIEVER"
    if event_name.startswith("on_chain_") and is_root:
        return "AGENT"
    return "NODE"


class LangGraphTargetAdapter:
    """Invoke a graph and normalize LangChain v2 stream events when available."""

    def __init__(self, spec: LangGraphTargetSpec) -> None:
        self._spec = spec
        self._graph: Any = None
        self._runtime = AgentQARuntime()
        self._factory_injection = False

    async def prepare(self, context: RunContext) -> None:
        del context
        entrypoint = load_entrypoint(self._spec.entrypoint)
        if hasattr(entrypoint, "invoke") or hasattr(entrypoint, "ainvoke"):
            self._graph = entrypoint
            return
        if not callable(entrypoint):
            raise TargetConfigurationError("LangGraph entrypoint is neither a graph nor a factory")

        signature = inspect.signature(entrypoint)
        if "agentqa_runtime" in signature.parameters:
            built = entrypoint(agentqa_runtime=self._runtime)
            self._factory_injection = True
        elif not signature.parameters:
            built = entrypoint()
        else:
            raise TargetConfigurationError(
                "LangGraph factory must accept no arguments or agentqa_runtime"
            )
        if inspect.isawaitable(built):
            built = await built
        if not any(hasattr(built, name) for name in ("invoke", "ainvoke", "astream_events")):
            raise TargetConfigurationError("LangGraph factory did not return an invokable graph")
        self._graph = built

    async def cleanup(self, context: RunContext) -> None:
        del context
        graph = self._graph
        self._graph = None
        if graph is None:
            return
        close = getattr(graph, "aclose", None)
        if callable(close):
            await close()

    def _input(self, test_input: TestInput) -> dict[str, Any]:
        state: dict[str, Any] = dict(test_input.initial_state)
        state.update(test_input.input)
        if test_input.messages:
            state[self._spec.input_key] = test_input.messages
        elif self._spec.input_key not in state:
            state[self._spec.input_key] = test_input.input
        return state

    @staticmethod
    def _config(context: RunContext) -> dict[str, Any]:
        return {
            "tags": ["agentqa", f"run:{context.run_id}", f"test:{context.test_case_id}"],
            "metadata": {
                "agentqa_run_id": context.run_id,
                "agentqa_test_id": context.test_case_id,
                "agentqa_trial_id": context.trial_id,
                "agentqa_trace_id": context.trace_id,
                "agentqa_mutation_id": context.mutation_id,
            },
        }

    def _output(self, value: Any) -> Any:
        normalized = to_jsonable(value)
        if self._spec.output_path is not None:
            return extract_path(normalized, self._spec.output_path)
        if isinstance(normalized, dict) and isinstance(normalized.get("messages"), list):
            messages = normalized["messages"]
            if messages:
                last = messages[-1]
                if isinstance(last, dict) and "content" in last:
                    return last["content"]
                return last
        return normalized

    def _state_diff(self, output: Any) -> dict[str, Any] | None:
        if not self._spec.state_diff_allowlist or not isinstance(output, dict):
            return None
        return {
            key: safe_jsonable(output[key])
            for key in self._spec.state_diff_allowlist
            if key in output
        }

    async def _event_stream(
        self,
        graph_input: dict[str, Any],
        config: dict[str, Any],
        context: RunContext,
    ) -> tuple[Any, int]:
        trace_id = context.trace_id or new_id()
        open_spans: dict[str, _OpenSpan] = {}
        event_count = 0
        final_output: Any = None
        span_context = trace.get_current_span().get_span_context()
        runner_span_id = f"{span_context.span_id:016x}" if span_context.is_valid else None
        trace_options = get_current_trace_options()
        stream = self._graph.astream_events(graph_input, config=config, version="v2")
        async for raw_event in stream:
            if not isinstance(raw_event, dict):
                continue
            event_name = str(raw_event.get("event") or "on_unknown")
            run_id = str(raw_event.get("run_id") or new_id())
            parents = raw_event.get("parent_ids")
            parent_ids = [str(value) for value in parents] if isinstance(parents, list) else []
            is_root = not parent_ids
            data = raw_event.get("data")
            data = data if isinstance(data, dict) else {}
            suffix = event_name.rsplit("_", 1)[-1]
            if suffix == "start":
                open_spans[run_id] = _OpenSpan(
                    span_id=run_id,
                    parent_span_id=parent_ids[-1] if parent_ids else runner_span_id,
                    kind=_event_kind(event_name, is_root=is_root),
                    name=str(raw_event.get("name") or event_name),
                    input=content_value(data.get("input"), trace_options),
                    started_at=utc_now(),
                    attributes={
                        "event": event_name,
                        "tags": safe_jsonable(raw_event.get("tags") or []),
                    },
                )
                continue
            if suffix not in {"end", "error"}:
                continue
            opened = open_spans.pop(run_id, None)
            if opened is None:
                opened = _OpenSpan(
                    span_id=run_id,
                    parent_span_id=parent_ids[-1] if parent_ids else runner_span_id,
                    kind=_event_kind(event_name, is_root=is_root),
                    name=str(raw_event.get("name") or event_name),
                    input=None,
                    started_at=utc_now(),
                    attributes={"event": event_name},
                )
            output = data.get("output")
            if suffix == "end" and is_root and output is not None:
                final_output = output
            error = None
            if suffix == "error":
                error = ErrorInfo(
                    code=TargetErrorCode.LANGGRAPH_RUNTIME_ERROR.value,
                    message="LangGraph event reported an error",
                )
            record_trace_event(
                TraceEvent(
                    trace_id=trace_id,
                    span_id=opened.span_id,
                    parent_span_id=opened.parent_span_id,
                    kind=opened.kind,
                    name=opened.name,
                    input=opened.input,
                    output=content_value(output, trace_options),
                    started_at=opened.started_at,
                    ended_at=utc_now(),
                    status="ERROR" if error is not None else "OK",
                    error=error,
                    attributes=opened.attributes,
                    state_diff=self._state_diff(output),
                )
            )
            event_count += 1
        current_store = get_current_trace_store()
        if current_store is not None and open_spans:
            current_store.add_diagnostic(
                trace_id,
                f"{len(open_spans)} LangGraph span(s) did not end",
            )
        return final_output, event_count

    async def _invoke_graph(
        self,
        graph_input: dict[str, Any],
        config: dict[str, Any],
        context: RunContext,
    ) -> tuple[Any, int]:
        if callable(getattr(self._graph, "astream_events", None)):
            return await self._event_stream(graph_input, config, context)
        if callable(getattr(self._graph, "ainvoke", None)):
            return await self._graph.ainvoke(graph_input, config=config), 0
        invoke = getattr(self._graph, "invoke", None)
        if callable(invoke):
            return await asyncio.to_thread(invoke, graph_input, config=config), 0
        raise TargetConfigurationError("LangGraph target is not invokable")

    async def invoke(self, test_input: TestInput, context: RunContext) -> TargetResult:
        started_at = utc_now()
        capabilities: dict[str, Any] = {
            **self._runtime.capabilities,
            "event_stream": callable(getattr(self._graph, "astream_events", None)),
            "factory_injection": self._factory_injection,
        }
        if not self._factory_injection:
            capabilities["injection_hint"] = (
                "export build_graph(agentqa_runtime=None) to enable tool and retriever wrapping"
            )
        if self._graph is None:
            return TargetResult(
                status="ERROR",
                error=ErrorInfo(
                    code=TargetErrorCode.LANGGRAPH_RUNTIME_ERROR.value,
                    message="LangGraph adapter has not been prepared",
                ),
                started_at=started_at,
                ended_at=utc_now(),
                metadata={"capabilities": capabilities},
            )
        try:
            output, trace_event_count = await self._invoke_graph(
                self._input(test_input), self._config(context), context
            )
            if output is None:
                raise ValueError("LangGraph invocation produced no final output")
            extracted = self._output(output)
        except (KeyError, ValueError) as exc:
            return TargetResult(
                status="ERROR",
                error=ErrorInfo(
                    code=TargetErrorCode.LANGGRAPH_OUTPUT_ERROR.value,
                    message=str(exc),
                ),
                started_at=started_at,
                ended_at=utc_now(),
                metadata={"capabilities": capabilities},
            )
        except Exception as exc:
            return TargetResult(
                status="ERROR",
                error=ErrorInfo(
                    code=TargetErrorCode.LANGGRAPH_RUNTIME_ERROR.value,
                    message=f"LangGraph invocation raised {type(exc).__name__}",
                ),
                started_at=started_at,
                ended_at=utc_now(),
                metadata={"capabilities": capabilities},
            )
        return TargetResult(
            status="OK",
            output=extracted,
            started_at=started_at,
            ended_at=utc_now(),
            metadata={
                "capabilities": capabilities,
                "trace_event_count": trace_event_count,
                "latency_ms": (utc_now() - started_at).total_seconds() * 1000,
            },
        )
