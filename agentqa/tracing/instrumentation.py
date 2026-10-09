"""Optional in-process OpenInference/OpenTelemetry instrumentation registry."""

import importlib
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.sdk.trace import TracerProvider

from agentqa.tracing.options import TraceOptions

_INSTRUMENTORS = {
    "openai": ("openinference.instrumentation.openai", "OpenAIInstrumentor"),
    "langchain": ("openinference.instrumentation.langchain", "LangChainInstrumentor"),
    "langgraph": ("openinference.instrumentation.langchain", "LangChainInstrumentor"),
    "httpx": ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
}


@dataclass(slots=True)
class InstrumentationManager:
    """Activate optional instrumentors and reverse only activations made here."""

    activated: list[Any] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)

    def activate(
        self,
        frameworks: list[str],
        tracer_provider: TracerProvider,
        trace_options: TraceOptions | None = None,
    ) -> None:
        """Activate each available framework instrumentor once."""
        seen: set[tuple[str, str]] = set()
        for framework in frameworks:
            specification = _INSTRUMENTORS.get(framework)
            if specification is None:
                self.unavailable.append(framework)
                continue
            if specification in seen:
                continue
            seen.add(specification)
            module_name, class_name = specification
            try:
                module = importlib.import_module(module_name)
                instrumentor = getattr(module, class_name)()
                instrumentor.instrument(tracer_provider=tracer_provider)
            except Exception:
                if framework == "openai":
                    try:
                        from agentqa.tracing.openai_fallback import OpenAIFallbackInstrumentor

                        instrumentor = OpenAIFallbackInstrumentor()
                        instrumentor.instrument(
                            tracer_provider=tracer_provider,
                            trace_options=trace_options,
                        )
                    except (ImportError, AttributeError):
                        self.unavailable.append(framework)
                        continue
                    self.activated.append(instrumentor)
                    continue
                self.unavailable.append(framework)
                continue
            self.activated.append(instrumentor)

    def close(self) -> None:
        """Uninstrument in reverse order without obscuring a run result."""
        for instrumentor in reversed(self.activated):
            try:
                instrumentor.uninstrument()
            except Exception:
                continue
        self.activated.clear()


def instrumentation_install_hint(framework: str) -> str:
    """Return a concrete optional-dependency hint for a missing instrumentor."""
    return (
        f"instrumentation for {framework!r} is unavailable; "
        "install it with: pip install 'agentqakit[instrumentation]'"
    )
