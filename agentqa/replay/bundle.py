"""Deterministic failure bundle creation and safe extraction."""

import platform
import stat
import sys
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml

from agentqa import __version__
from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    MutationSpec,
    RunContext,
    TargetSpec,
    TestResult,
    TraceRecord,
)
from agentqa.models.specs import TrialPolicy
from agentqa.replay.models import CassetteEntry
from agentqa.replay.normalization import safe_value
from agentqa.replay.store import CassetteFormatError
from agentqa.specs import normalize_spec
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.storage import redact_text_secrets

FAILURE_BUNDLE_SCHEMA_VERSION = "agentqa.failure-bundle/v1alpha1"
_MAX_BUNDLE_BYTES = 64 * 1024 * 1024
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _safe_mapping(value: Any) -> dict[str, Any]:
    normalized = safe_value(value)
    if not isinstance(normalized, dict):
        raise ValueError("failure bundle document must be a mapping")
    return cast(dict[str, Any], normalized)


def _yaml(value: Any) -> bytes:
    return yaml.safe_dump(
        _safe_mapping(value),
        allow_unicode=True,
        sort_keys=True,
    ).encode("utf-8")


def _json(value: Any) -> bytes:
    return canonical_json_bytes(_safe_mapping(value)) + b"\n"


def _cassette_files(cassette_path: Path) -> dict[str, bytes]:
    try:
        cassette = cassette_path.read_bytes()
    except OSError as exc:
        raise CassetteFormatError(f"could not read trial cassette: {type(exc).__name__}") from None
    files = {"cassette.jsonl": cassette}
    try:
        expected_cassette = (
            cassette_path.with_name(f"{cassette_path.name}.sha256")
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        raise CassetteFormatError("trial cassette checksum is missing") from None
    if expected_cassette != sha256_digest(cassette):
        raise CassetteFormatError("trial cassette checksum mismatch")
    for line_number, line in enumerate(cassette.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = CassetteEntry.model_validate_json(line)
            reference = entry.result_ref
        except ValueError:
            raise CassetteFormatError(f"invalid trial cassette at line {line_number}") from None
        if not entry.verify_checksum():
            raise CassetteFormatError(
                f"trial cassette entry checksum mismatch at line {line_number}"
            )
        if reference is None:
            continue
        relative = PurePosixPath(str(reference))
        if relative.is_absolute() or ".." in relative.parts:
            raise CassetteFormatError("cassette blob reference is unsafe")
        source = (cassette_path.parent / Path(*relative.parts)).resolve()
        try:
            source.relative_to(cassette_path.parent.resolve())
            blob = source.read_bytes()
            expected = source.with_name(f"{source.name}.sha256").read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            raise CassetteFormatError("cassette blob is missing or unsafe") from None
        if expected != sha256_digest(blob):
            raise CassetteFormatError("cassette blob checksum mismatch")
        files[relative.as_posix()] = blob
        files[f"{relative.as_posix()}.sha256"] = f"{expected}\n".encode()
    return files


def _readme(
    result: TestResult,
    *,
    original_command: str | None,
) -> bytes:
    violations = ", ".join(item.type for item in result.violations) or "none"
    expected = "The target and every configured evaluator should pass."
    actual = f"Target status: {result.target_result.status}; violation types: {violations}."
    command = redact_text_secrets(original_command or "not captured (programmatic run)")
    text = f"""# AgentQAKit failure reproduction

Test case: `{result.test_case_id}`  
Trial: `{result.trial_id}`  

## Failure summary

{actual}

## Original command

```text
{command}
```

## Replay command

```text
agentqa replay failure-{result.trial_id}.zip
```

## Expected behavior

{expected}

## Actual behavior

{actual}

## Sensitive information

Specifications, interaction arguments, results, errors, and environment data
were normalized and redacted before this bundle and its checksums were created.
Strict replay does not access live dependencies unless explicitly allowed.
"""
    return text.encode("utf-8")


def build_failure_bundle(
    *,
    context: RunContext,
    test_spec: AgentTestSpec,
    target_spec: TargetSpec,
    mutation_spec: MutationSpec | None,
    manifest: AgentManifest | None,
    result: TestResult,
    trace: TraceRecord,
    cassette_path: Path,
    original_command: str | None,
) -> bytes:
    """Build one deterministic, redacted ZIP for a failed trial."""
    test_document = normalize_spec(test_spec)
    test_document["trials"] = 1
    test_document["trialPolicy"] = normalize_spec(TrialPolicy(strategy="all"))
    bundle_manifest = {
        "schema_version": FAILURE_BUNDLE_SCHEMA_VERSION,
        "run_id": context.run_id,
        "test_case_id": result.test_case_id,
        "trial_id": result.trial_id,
        "trace_id": result.trace_id,
        "random_seed": context.random_seed,
        "trial_seed": result.trial_seed,
        "has_mutation": mutation_spec is not None,
        "has_agent_manifest": manifest is not None,
        "cassette": "cassette.jsonl",
        "target": "target.yaml",
        "test": "test.yaml",
        "mutation": "mutation.yaml" if mutation_spec is not None else None,
    }
    files: dict[str, bytes] = {
        "manifest.yaml": _yaml(bundle_manifest),
        "target.yaml": _yaml(normalize_spec(target_spec)),
        "test.yaml": _yaml(test_document),
        "result.json": _json(result.model_dump(mode="json")),
        "trace.json": _json(trace.model_dump(mode="json")),
        "environment.json": _json(
            {
                "agentqakit_version": __version__,
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": sys.platform,
            }
        ),
        "README.md": _readme(result, original_command=original_command),
    }
    if mutation_spec is not None:
        files["mutation.yaml"] = _yaml(normalize_spec(mutation_spec))
    if manifest is not None:
        files["agent-manifest.yaml"] = _yaml(normalize_spec(manifest))
    files.update(_cassette_files(cassette_path))
    checksums = "".join(
        f"{sha256_digest(payload)}  {name}\n" for name, payload in sorted(files.items())
    ).encode("utf-8")
    files["checksums.txt"] = checksums

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            info = zipfile.ZipInfo(name, _ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def _safe_member(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    return (
        bool(info.filename)
        and bool(path.parts)
        and "\\" not in info.filename
        and not path.is_absolute()
        and ".." not in path.parts
        and not stat.S_ISLNK(mode)
        and not info.is_dir()
    )


def extract_failure_bundle(bundle: Path, destination: Path) -> dict[str, Any]:
    """Verify and safely extract a failure bundle into an empty directory."""
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise CassetteFormatError("bundle extraction directory must be empty")
    try:
        with zipfile.ZipFile(bundle) as archive:
            infos = archive.infolist()
            if any(not _safe_member(info) for info in infos):
                raise CassetteFormatError("failure bundle contains an unsafe path")
            if len({info.filename for info in infos}) != len(infos):
                raise CassetteFormatError("failure bundle contains duplicate paths")
            if sum(info.file_size for info in infos) > _MAX_BUNDLE_BYTES:
                raise CassetteFormatError("failure bundle exceeds the extraction limit")
            payloads = {info.filename: archive.read(info) for info in infos}
    except (OSError, zipfile.BadZipFile):
        raise CassetteFormatError("failure bundle is not a readable ZIP") from None
    required = {
        "manifest.yaml",
        "target.yaml",
        "test.yaml",
        "result.json",
        "trace.json",
        "cassette.jsonl",
        "environment.json",
        "README.md",
        "checksums.txt",
    }
    if not required.issubset(payloads):
        raise CassetteFormatError("failure bundle is missing required files")
    try:
        checksum_lines = payloads["checksums.txt"].decode("utf-8").splitlines()
        pairs = [line.split("  ", 1) for line in checksum_lines]
        expected = {name: digest for digest, name in pairs}
    except (UnicodeDecodeError, ValueError):
        raise CassetteFormatError("failure bundle checksums are invalid") from None
    if len(expected) != len(checksum_lines):
        raise CassetteFormatError("failure bundle checksums contain duplicate paths")
    covered = set(payloads) - {"checksums.txt"}
    if set(expected) != covered:
        raise CassetteFormatError("failure bundle checksum coverage is incomplete")
    for name in covered:
        if expected[name] != sha256_digest(payloads[name]):
            raise CassetteFormatError(f"failure bundle checksum mismatch for {name}")
    try:
        manifest = yaml.safe_load(payloads["manifest.yaml"])
    except yaml.YAMLError:
        raise CassetteFormatError("failure bundle manifest is invalid") from None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != (
        FAILURE_BUNDLE_SCHEMA_VERSION
    ):
        raise CassetteFormatError("failure bundle manifest version is unsupported")
    if bool(manifest.get("has_mutation")) != ("mutation.yaml" in payloads):
        raise CassetteFormatError("failure bundle mutation metadata is inconsistent")
    if bool(manifest.get("has_agent_manifest")) != ("agent-manifest.yaml" in payloads):
        raise CassetteFormatError("failure bundle agent manifest metadata is inconsistent")
    for name, payload in payloads.items():
        target = destination / Path(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return cast(dict[str, Any], manifest)
