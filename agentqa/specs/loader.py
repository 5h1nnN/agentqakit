"""Safe YAML loading with source locations and strict model validation."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from agentqa.errors import AgentQAError
from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    GenerationGoal,
    MutationSpec,
    QualityGateSpec,
    TargetSpec,
)
from agentqa.models.specs import HttpTargetSpec, LangGraphTargetSpec, PythonTargetSpec
from agentqa.specs.hashing import hash_spec, hash_spec_collection
from agentqa.specs.migrations import UnsupportedSpecVersionError, migrate_document
from agentqa.specs.rendering import TemplateRenderError, render_templates

SpecDocument: TypeAlias = (
    AgentTestSpec
    | HttpTargetSpec
    | PythonTargetSpec
    | LangGraphTargetSpec
    | MutationSpec
    | AgentManifest
    | QualityGateSpec
    | GenerationGoal
)
LocationPath: TypeAlias = tuple[str | int, ...]

_TARGET_ADAPTER: TypeAdapter[Any] = TypeAdapter(cast(Any, TargetSpec))
_MODEL_BY_KIND: dict[str, type[BaseModel] | TypeAdapter[Any]] = {
    "AgentTest": AgentTestSpec,
    "Target": _TARGET_ADAPTER,
    "Mutation": MutationSpec,
    "AgentManifest": AgentManifest,
    "QualityGate": QualityGateSpec,
    "GenerationGoal": GenerationGoal,
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclass(frozen=True, slots=True)
class SourcePosition:
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class SpecIssue:
    """One safe validation diagnostic."""

    source: Path
    document_index: int
    path: LocationPath
    message: str
    code: str
    position: SourcePosition | None = None

    def format(self) -> str:
        location = str(self.source)
        if self.position is not None:
            location += f":{self.position.line}:{self.position.column}"
        document = f" document {self.document_index}"
        field = ".".join(str(part) for part in self.path)
        suffix = f" [{field}]" if field else ""
        return f"{location}:{document}{suffix}: {self.message}"


class SpecLoadError(AgentQAError):
    """One or more YAML or model validation errors."""

    def __init__(self, issues: list[SpecIssue]) -> None:
        super().__init__(
            f"spec validation failed with {len(issues)} issue(s)",
            code="SPEC_ERROR",
            details={"issue_count": len(issues)},
        )
        self.issues = issues


@dataclass(frozen=True, slots=True)
class LoadedSpec:
    """A validated spec plus source and reproducibility metadata."""

    source: Path
    document_index: int
    spec: SpecDocument
    spec_hash: str
    redacted_document: dict[str, Any]


def _collect_marks(
    node: Node,
    path: LocationPath,
    marks: dict[LocationPath, SourcePosition],
) -> None:
    marks[path] = SourcePosition(node.start_mark.line + 1, node.start_mark.column + 1)
    if isinstance(node, MappingNode):
        for key_node, value_node in node.value:
            if isinstance(key_node, ScalarNode):
                key = key_node.value
                _collect_marks(value_node, (*path, key), marks)
    elif isinstance(node, SequenceNode):
        for index, child in enumerate(node.value):
            _collect_marks(child, (*path, index), marks)


def _position_for(
    path: LocationPath,
    marks: Mapping[LocationPath, SourcePosition],
) -> SourcePosition | None:
    candidates = [path]
    if path and isinstance(path[0], str) and path[0] in {"http", "python", "langgraph"}:
        candidates.append(path[1:])
    for candidate in candidates:
        current = candidate
        while current:
            if current in marks:
                return marks[current]
            current = current[:-1]
    return marks.get(())


def _validation_issues(
    exc: ValidationError,
    *,
    source: Path,
    document_index: int,
    marks: Mapping[LocationPath, SourcePosition],
) -> list[SpecIssue]:
    issues: list[SpecIssue] = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        path = cast(LocationPath, tuple(error["loc"]))
        if path and path[0] in {"http", "python", "langgraph"}:
            path = path[1:]
        issues.append(
            SpecIssue(
                source=source,
                document_index=document_index,
                path=path,
                message=str(error["msg"]),
                code=str(error["type"]),
                position=_position_for(path, marks),
            )
        )
    return issues


def _parse_yaml(source: Path) -> tuple[list[Any], list[dict[LocationPath, SourcePosition]]]:
    text = source.read_text(encoding="utf-8")
    try:
        documents = list(yaml.load_all(text, Loader=_UniqueKeySafeLoader))
        nodes = list(yaml.compose_all(text, Loader=yaml.SafeLoader))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        position = None if mark is None else SourcePosition(mark.line + 1, mark.column + 1)
        message = getattr(exc, "problem", None) or "invalid YAML"
        raise SpecLoadError(
            [
                SpecIssue(
                    source=source,
                    document_index=1,
                    path=(),
                    message=str(message),
                    code="yaml_error",
                    position=position,
                )
            ]
        ) from None

    marks_by_document: list[dict[LocationPath, SourcePosition]] = []
    for node in nodes:
        marks: dict[LocationPath, SourcePosition] = {}
        if node is not None:
            _collect_marks(node, (), marks)
        marks_by_document.append(marks)
    return documents, marks_by_document


def _validate_document(
    document: Any,
    *,
    source: Path,
    document_index: int,
    marks: Mapping[LocationPath, SourcePosition],
    environment: Mapping[str, str] | None,
) -> LoadedSpec:
    if not isinstance(document, dict):
        raise SpecLoadError(
            [
                SpecIssue(
                    source=source,
                    document_index=document_index,
                    path=(),
                    message="document must be a mapping",
                    code="mapping_type",
                    position=marks.get(()),
                )
            ]
        )

    try:
        rendered = render_templates(
            document,
            environment=environment,
            resolve_input=False,
        )
    except TemplateRenderError as exc:
        raise SpecLoadError(
            [
                SpecIssue(
                    source=source,
                    document_index=document_index,
                    path=exc.path,
                    message=exc.message,
                    code=exc.code,
                    position=_position_for(exc.path, marks),
                )
            ]
        ) from None

    try:
        migrated = migrate_document(cast(dict[str, Any], rendered.value))
    except UnsupportedSpecVersionError as exc:
        path = ("apiVersion",)
        raise SpecLoadError(
            [
                SpecIssue(
                    source=source,
                    document_index=document_index,
                    path=path,
                    message=str(exc),
                    code="unsupported_version",
                    position=_position_for(path, marks),
                )
            ]
        ) from None

    kind = migrated.get("kind")
    validator = _MODEL_BY_KIND.get(kind) if isinstance(kind, str) else None
    if validator is None:
        path = ("kind",)
        raise SpecLoadError(
            [
                SpecIssue(
                    source=source,
                    document_index=document_index,
                    path=path,
                    message=f"unknown or missing spec kind: {kind!r}",
                    code="unknown_kind",
                    position=_position_for(path, marks),
                )
            ]
        )

    try:
        if isinstance(validator, TypeAdapter):
            spec = validator.validate_python(migrated)
        else:
            spec = validator.model_validate(migrated)
    except ValidationError as exc:
        raise SpecLoadError(
            _validation_issues(
                exc,
                source=source,
                document_index=document_index,
                marks=marks,
            )
        ) from None

    redacted = cast(dict[str, Any], rendered.redacted)
    return LoadedSpec(
        source=source,
        document_index=document_index,
        spec=cast(SpecDocument, spec),
        spec_hash=hash_spec(spec),
        redacted_document=redacted,
    )


def load_spec_file(
    path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> list[LoadedSpec]:
    """Load and validate every non-empty YAML document in one file."""
    source = Path(path).resolve()
    documents, marks_by_document = _parse_yaml(source)
    loaded: list[LoadedSpec] = []
    issues: list[SpecIssue] = []
    for index, document in enumerate(documents, start=1):
        if document is None:
            continue
        marks = marks_by_document[index - 1] if index <= len(marks_by_document) else {}
        try:
            loaded.append(
                _validate_document(
                    document,
                    source=source,
                    document_index=index,
                    marks=marks,
                    environment=environment,
                )
            )
        except SpecLoadError as exc:
            issues.extend(exc.issues)
    if issues:
        raise SpecLoadError(issues)
    return loaded


def discover_spec_files(path: str | Path) -> list[Path]:
    """Discover YAML files in deterministic relative-path order."""
    source = Path(path).resolve()
    if source.is_file():
        if source.suffix.lower() not in {".yaml", ".yml"}:
            raise SpecLoadError(
                [SpecIssue(source, 1, (), "expected a .yaml or .yml file", "file_type")]
            )
        return [source]
    if not source.is_dir():
        raise SpecLoadError([SpecIssue(source, 1, (), "path does not exist", "path_missing")])
    files = sorted(
        (
            candidate
            for candidate in source.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in {".yaml", ".yml"}
            and not any(part.startswith(".") for part in candidate.relative_to(source).parts)
        ),
        key=lambda candidate: candidate.relative_to(source).as_posix(),
    )
    if not files:
        raise SpecLoadError([SpecIssue(source, 1, (), "directory contains no YAML files", "empty")])
    return files


def load_specs(
    path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> list[LoadedSpec]:
    """Load a file or directory, collecting diagnostics from every file."""
    files = discover_spec_files(path)
    loaded: list[LoadedSpec] = []
    issues: list[SpecIssue] = []
    for source in files:
        try:
            loaded.extend(load_spec_file(source, environment=environment))
        except SpecLoadError as exc:
            issues.extend(exc.issues)
    if issues:
        raise SpecLoadError(issues)
    return loaded


def loaded_collection_hash(specs: list[LoadedSpec]) -> str:
    """Return a stable hash for specs in their deterministic load order."""
    return hash_spec_collection([item.spec_hash for item in specs])


def iter_formatted_issues(exc: SpecLoadError) -> Iterator[str]:
    """Yield user-safe diagnostics without embedding rejected values."""
    for issue in exc.issues:
        yield issue.format()
