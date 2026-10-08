"""Agent test specification loading and validation."""

from agentqa.specs.hashing import hash_spec, hash_spec_collection, normalize_spec
from agentqa.specs.loader import (
    LoadedSpec,
    SpecIssue,
    SpecLoadError,
    discover_spec_files,
    load_spec_file,
    load_specs,
    loaded_collection_hash,
)
from agentqa.specs.rendering import RenderedTemplate, TemplateRenderError, render_templates
from agentqa.specs.schema import build_json_schemas, export_json_schemas

__all__ = [
    "LoadedSpec",
    "RenderedTemplate",
    "SpecIssue",
    "SpecLoadError",
    "TemplateRenderError",
    "build_json_schemas",
    "discover_spec_files",
    "export_json_schemas",
    "hash_spec",
    "hash_spec_collection",
    "load_spec_file",
    "load_specs",
    "loaded_collection_hash",
    "normalize_spec",
    "render_templates",
]
