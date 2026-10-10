"""AgentQAKit public package."""

from importlib.metadata import PackageNotFoundError, version

from agentqa.sdk import agent, retriever, tool

try:
    __version__ = version("agentqakit")
except PackageNotFoundError:
    __version__ = "0.1.0a0"

__all__ = ["__version__", "agent", "retriever", "tool"]
