import pytest

from agentqa.models import HttpTargetSpec
from agentqa.targets import HttpTargetAdapter, TargetAdapterRegistry, create_target_adapter
from agentqa.targets.errors import TargetConfigurationError


def _http_spec() -> HttpTargetSpec:
    return HttpTargetSpec.model_validate(
        {
            "metadata": {"name": "local"},
            "type": "http",
            "url": "http://localhost:8000/chat",
        }
    )


def test_default_registry_creates_matching_adapter() -> None:
    assert isinstance(create_target_adapter(_http_spec()), HttpTargetAdapter)


def test_registry_rejects_duplicate_registration() -> None:
    registry = TargetAdapterRegistry()
    registry.register("http", lambda spec: HttpTargetAdapter(_http_spec()))

    with pytest.raises(TargetConfigurationError, match="already registered"):
        registry.register("http", lambda spec: HttpTargetAdapter(_http_spec()))
