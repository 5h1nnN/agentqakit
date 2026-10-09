"""Target adapter registry and built-in factories."""

from collections.abc import Callable

from agentqa.models import HttpTargetSpec, LangGraphTargetSpec, PythonTargetSpec, TargetSpec
from agentqa.targets.base import TargetAdapter
from agentqa.targets.errors import TargetConfigurationError
from agentqa.targets.http import HttpTargetAdapter
from agentqa.targets.langgraph import LangGraphTargetAdapter
from agentqa.targets.python import PythonTargetAdapter

AdapterFactory = Callable[[TargetSpec], TargetAdapter]


class TargetAdapterRegistry:
    """Map strict target type names to adapter constructors."""

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, target_type: str, factory: AdapterFactory) -> None:
        if not target_type:
            raise TargetConfigurationError("target adapter type cannot be empty")
        if target_type in self._factories:
            raise TargetConfigurationError(f"target adapter {target_type!r} is already registered")
        self._factories[target_type] = factory

    def create(self, spec: TargetSpec) -> TargetAdapter:
        factory = self._factories.get(spec.type)
        if factory is None:
            raise TargetConfigurationError(f"no adapter registered for target type {spec.type!r}")
        return factory(spec)


def _http_factory(spec: TargetSpec) -> TargetAdapter:
    if not isinstance(spec, HttpTargetSpec):
        raise TargetConfigurationError("http adapter received an incompatible target spec")
    return HttpTargetAdapter(spec)


def _python_factory(spec: TargetSpec) -> TargetAdapter:
    if not isinstance(spec, PythonTargetSpec):
        raise TargetConfigurationError("python adapter received an incompatible target spec")
    return PythonTargetAdapter(spec)


def _langgraph_factory(spec: TargetSpec) -> TargetAdapter:
    if not isinstance(spec, LangGraphTargetSpec):
        raise TargetConfigurationError("langgraph adapter received an incompatible target spec")
    return LangGraphTargetAdapter(spec)


default_registry = TargetAdapterRegistry()
default_registry.register("http", _http_factory)
default_registry.register("python", _python_factory)
default_registry.register("langgraph", _langgraph_factory)


def create_target_adapter(spec: TargetSpec) -> TargetAdapter:
    """Build one adapter from the process-wide built-in registry."""
    return default_registry.create(spec)
