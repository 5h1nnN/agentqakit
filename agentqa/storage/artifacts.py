"""Crash-safe local artifact directory and JSON/JSONL writer."""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from agentqa.errors import AgentQAError
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.storage.redaction import redact_secrets

ARTIFACT_SCHEMA_VERSION = "agentqa.artifact/v1alpha1"
_RUN_DIRECTORIES = ("traces", "cassettes", "failures")


class ArtifactError(AgentQAError):
    """Raised when artifact persistence or verification fails."""

    def __init__(self, message: str, *, code: str = "ARTIFACT_ERROR") -> None:
        super().__init__(message, code=code)


class ArtifactWriter:
    """Own one run directory and persist records using atomic replacement."""

    def __init__(self, base_directory: str | Path, run_id: str) -> None:
        if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
            raise ArtifactError("run_id must be one non-empty path component")
        self.root = Path(base_directory).resolve() / "runs" / run_id
        self._lock = threading.RLock()
        self._finalized = (self.root / "summary.json").exists()

    def initialize(self, run: Any | None = None) -> Path:
        """Create the run layout and optionally persist initial run metadata."""
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            for name in _RUN_DIRECTORIES:
                (self.root / name).mkdir(exist_ok=True)
            results = self.root / "results.jsonl"
            if not results.exists():
                self._write_with_checksum(results, b"")
            if run is not None:
                self.write_run(run)
        return self.root

    def _ensure_open(self) -> None:
        if self._finalized:
            raise ArtifactError("run artifacts are already finalized", code="ARTIFACT_FINALIZED")

    def _resolve(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ArtifactError("artifact path must be relative", code="ARTIFACT_PATH_ERROR")
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            raise ArtifactError(
                "artifact path escapes the run directory",
                code="ARTIFACT_PATH_ERROR",
            ) from None
        return candidate

    @staticmethod
    def _replace_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            # POSIX permits opening a directory so the rename itself can be
            # flushed. Windows does not, and the file contents have already
            # been flushed above before the atomic replacement.
            if os.name != "nt":
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _write_with_checksum(self, path: Path, payload: bytes) -> str:
        digest = sha256_digest(payload)
        self._replace_bytes(path, payload)
        checksum_path = path.with_name(f"{path.name}.sha256")
        self._replace_bytes(checksum_path, f"{digest}\n".encode())
        return digest

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", by_alias=False)
        if not isinstance(value, dict):
            raise ArtifactError("JSON artifact payload must be a mapping")
        return cast(dict[str, Any], redact_secrets(value))

    @staticmethod
    def _document_payload(value: Any) -> bytes:
        document = ArtifactWriter._as_mapping(value)
        existing_version = document.get("schema_version")
        if existing_version not in {None, ARTIFACT_SCHEMA_VERSION}:
            raise ArtifactError("artifact has an incompatible schema_version")
        document["schema_version"] = ARTIFACT_SCHEMA_VERSION
        return canonical_json_bytes(document) + b"\n"

    def _write_json(self, relative_path: str | Path, value: Any) -> str:
        self._ensure_open()
        path = self._resolve(relative_path)
        return self._write_with_checksum(path, self._document_payload(value))

    def write_run(self, run: Any) -> str:
        """Write ``run.json``."""
        with self._lock:
            return self._write_json("run.json", run)

    def write_resolved_target(self, target: Any) -> str:
        """Write the redacted resolved target configuration."""
        with self._lock:
            return self._write_json("resolved-target.json", target)

    def write_resolved_tests(self, tests: Any) -> str:
        """Write the redacted resolved test collection."""
        with self._lock:
            if isinstance(tests, list):
                tests = {"tests": tests}
            return self._write_json("resolved-tests.json", tests)

    def write_resolved_mutations(self, mutations: Any) -> str:
        """Write the redacted mutation collection used by the run."""
        with self._lock:
            if isinstance(mutations, list):
                mutations = {"mutations": mutations}
            return self._write_json("resolved-mutations.json", mutations)

    def write_mutation_score(self, score: Any) -> str:
        """Write a recomputable mutation score report."""
        with self._lock:
            return self._write_json("mutation-score.json", score)

    def write_trace(self, trace_id: str, trace: Any) -> str:
        """Write one trace document under ``traces/``."""
        if Path(trace_id).name != trace_id or not trace_id:
            raise ArtifactError("trace_id must be one non-empty path component")
        with self._lock:
            return self._write_json(Path("traces") / f"{trace_id}.json", trace)

    def write_failure_bundle(self, trial_id: str, payload: bytes) -> tuple[str, str]:
        """Write one reproduction ZIP under ``failures/`` with a checksum sidecar."""
        if Path(trial_id).name != trial_id or not trial_id:
            raise ArtifactError("trial_id must be one non-empty path component")
        with self._lock:
            self._ensure_open()
            relative = Path("failures") / f"failure-{trial_id}.zip"
            digest = self._write_with_checksum(self._resolve(relative), payload)
            return relative.as_posix(), digest

    def append_result(self, result: Any) -> str:
        """Atomically append a checksummed record to ``results.jsonl``."""
        with self._lock:
            self._ensure_open()
            result_data = self._as_mapping(result)
            payload_digest = sha256_digest(canonical_json_bytes(result_data))
            record = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "payload_sha256": payload_digest,
                "data": result_data,
            }
            line = canonical_json_bytes(record) + b"\n"
            path = self._resolve("results.jsonl")
            current = path.read_bytes() if path.exists() else b""
            if current and not current.endswith(b"\n"):
                raise ArtifactError("results.jsonl has an incomplete final record")
            return self._write_with_checksum(path, current + line)

    def finalize_summary(self, summary: Any) -> str:
        """Write ``summary.json`` last and prevent later mutations."""
        with self._lock:
            self._ensure_open()
            digest = self._write_json("summary.json", summary)
            self._finalized = True
            return digest

    def verify_checksum(self, relative_path: str | Path) -> bool:
        """Verify the SHA-256 sidecar for one artifact."""
        path = self._resolve(relative_path)
        checksum_path = path.with_name(f"{path.name}.sha256")
        if not path.is_file() or not checksum_path.is_file():
            return False
        expected = checksum_path.read_text(encoding="utf-8").strip()
        return expected == sha256_digest(path.read_bytes())

    def read_result_records(self) -> list[dict[str, Any]]:
        """Read and verify all JSONL record payloads."""
        path = self._resolve("results.jsonl")
        if not self.verify_checksum("results.jsonl"):
            raise ArtifactError("results.jsonl checksum mismatch", code="ARTIFACT_CHECKSUM_ERROR")
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                record = json.loads(line)
                data = record["data"]
                expected = record["payload_sha256"]
            except (json.JSONDecodeError, KeyError, TypeError):
                raise ArtifactError(
                    f"invalid results.jsonl record at line {line_number}",
                    code="ARTIFACT_FORMAT_ERROR",
                ) from None
            if expected != sha256_digest(canonical_json_bytes(data)):
                raise ArtifactError(
                    f"results.jsonl payload checksum mismatch at line {line_number}",
                    code="ARTIFACT_CHECKSUM_ERROR",
                )
            records.append(data)
        return records
