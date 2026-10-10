"""Crash-safe canonical JSONL cassette storage."""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import JsonValue, ValidationError

from agentqa.conventions import utc_now
from agentqa.errors import AgentQAError
from agentqa.replay.models import CassetteEntry, CassetteError
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest


class ReplayError(AgentQAError):
    """Base class for safe record/replay failures."""

    def __init__(self, message: str, *, code: str = "REPLAY_ERROR") -> None:
        super().__init__(message, code=code)


class CassetteFormatError(ReplayError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="CASSETTE_FORMAT_ERROR")


class CassetteStore(Protocol):
    """Storage operations required by the replay engine."""

    def lookup(
        self,
        kind: Literal["tool", "retriever", "http"],
        name: str,
        arguments: JsonValue,
        call_index: int,
    ) -> CassetteEntry | None: ...

    def record_success(
        self,
        *,
        kind: Literal["tool", "retriever", "http"],
        name: str,
        arguments: JsonValue,
        call_index: int,
        result: JsonValue,
        trace_id: str,
        span_id: str,
    ) -> CassetteEntry: ...

    def record_error(
        self,
        *,
        kind: Literal["tool", "retriever", "http"],
        name: str,
        arguments: JsonValue,
        call_index: int,
        error: CassetteError,
        trace_id: str,
        span_id: str,
    ) -> CassetteEntry: ...

    def result(self, entry: CassetteEntry) -> JsonValue: ...

    def flush(self) -> None: ...


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class JsonlCassetteStore:
    """Read an immutable source cassette and atomically flush observed entries."""

    def __init__(
        self,
        *,
        source_path: Path | None = None,
        output_path: Path | None = None,
        max_inline_result_bytes: int = 65_536,
    ) -> None:
        if max_inline_result_bytes <= 0:
            raise ValueError("max_inline_result_bytes must be positive")
        self.source_path = None if source_path is None else source_path.resolve()
        self.output_path = None if output_path is None else output_path.resolve()
        if self.source_path is not None and self.source_path == self.output_path:
            raise ValueError("source and output cassette paths must differ")
        self.max_inline_result_bytes = max_inline_result_bytes
        self._lock = threading.RLock()
        self._source_entries: dict[tuple[str, str, bytes, int], CassetteEntry] = {}
        self._recorded_entries: list[CassetteEntry] = []
        if self.source_path is not None:
            self._load_source()

    @staticmethod
    def _key(
        kind: str,
        name: str,
        arguments: JsonValue,
        call_index: int,
    ) -> tuple[str, str, bytes, int]:
        return kind, name, canonical_json_bytes(arguments), call_index

    def _load_source(self) -> None:
        assert self.source_path is not None
        if not self.source_path.is_file():
            raise CassetteFormatError("source cassette does not exist")
        try:
            lines = self.source_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise CassetteFormatError(
                f"could not read source cassette: {type(exc).__name__}"
            ) from None
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                entry = CassetteEntry.model_validate_json(line)
            except ValidationError:
                raise CassetteFormatError(f"invalid cassette entry at line {line_number}") from None
            if not entry.verify_checksum():
                raise CassetteFormatError(f"cassette entry checksum mismatch at line {line_number}")
            key = self._key(entry.kind, entry.name, entry.arguments, entry.call_index)
            if key in self._source_entries:
                raise CassetteFormatError(f"duplicate cassette match key at line {line_number}")
            self._source_entries[key] = entry

    @staticmethod
    def _with_checksum(entry: CassetteEntry) -> CassetteEntry:
        checksum = sha256_digest(entry.checksum_payload())
        return entry.model_copy(update={"checksum": checksum})

    def _blob(self, result: JsonValue) -> tuple[JsonValue, str | None]:
        payload = canonical_json_bytes(result) + b"\n"
        if len(payload) <= self.max_inline_result_bytes:
            return result, None
        if self.output_path is None:
            raise ReplayError("large replay result requires an output cassette")
        digest = sha256_digest(payload)
        relative = Path("blobs") / f"{digest.removeprefix('sha256:')}.json"
        path = self.output_path.parent / relative
        _atomic_write(path, payload)
        _atomic_write(path.with_name(f"{path.name}.sha256"), f"{digest}\n".encode())
        return None, relative.as_posix()

    def lookup(
        self,
        kind: Literal["tool", "retriever", "http"],
        name: str,
        arguments: JsonValue,
        call_index: int,
    ) -> CassetteEntry | None:
        with self._lock:
            return self._source_entries.get(self._key(kind, name, arguments, call_index))

    def record_success(
        self,
        *,
        kind: Literal["tool", "retriever", "http"],
        name: str,
        arguments: JsonValue,
        call_index: int,
        result: JsonValue,
        trace_id: str,
        span_id: str,
    ) -> CassetteEntry:
        recorded_at = utc_now()
        inline, result_ref = self._blob(result)
        values: dict[str, object] = {
            "kind": kind,
            "name": name,
            "arguments": arguments,
            "call_index": call_index,
            "status": "success",
            "recorded_at": recorded_at,
            "trace_id": trace_id,
            "span_id": span_id,
            "checksum": "pending",
        }
        if result_ref is None:
            values["result"] = inline
        else:
            values["result_ref"] = result_ref
        entry = self._with_checksum(CassetteEntry.model_validate(values))
        with self._lock:
            self._recorded_entries.append(entry)
        return entry

    def import_entry(self, entry: CassetteEntry) -> None:
        """Import a verified entry observed by an isolated Python worker."""
        if not entry.verify_checksum():
            raise CassetteFormatError("imported cassette entry checksum mismatch")
        with self._lock:
            self._recorded_entries.append(entry)

    def record_error(
        self,
        *,
        kind: Literal["tool", "retriever", "http"],
        name: str,
        arguments: JsonValue,
        call_index: int,
        error: CassetteError,
        trace_id: str,
        span_id: str,
    ) -> CassetteEntry:
        entry = self._with_checksum(
            CassetteEntry(
                kind=kind,
                name=name,
                arguments=arguments,
                call_index=call_index,
                status="error",
                error=error,
                recorded_at=utc_now(),
                trace_id=trace_id,
                span_id=span_id,
                checksum="pending",
            )
        )
        with self._lock:
            self._recorded_entries.append(entry)
        return entry

    def result(self, entry: CassetteEntry) -> JsonValue:
        if entry.status != "success":
            raise ReplayError("error cassette entry has no result")
        if entry.result_ref is None:
            return entry.result
        if self.source_path is None:
            base = self.output_path.parent if self.output_path is not None else None
        else:
            base = self.source_path.parent
        if base is None:
            raise CassetteFormatError("cassette blob has no storage root")
        relative = Path(entry.result_ref)
        if relative.is_absolute():
            raise CassetteFormatError("cassette blob reference must be relative")
        path = (base / relative).resolve()
        try:
            path.relative_to(base.resolve())
        except ValueError:
            raise CassetteFormatError("cassette blob reference escapes its root") from None
        try:
            raw = path.read_bytes()
            expected = path.with_name(f"{path.name}.sha256").read_text(encoding="utf-8").strip()
            if expected != sha256_digest(raw):
                raise CassetteFormatError("cassette blob checksum mismatch")
            return cast(JsonValue, json.loads(raw))
        except (OSError, json.JSONDecodeError):
            raise CassetteFormatError("could not read cassette result blob") from None

    def flush(self) -> None:
        if self.output_path is None:
            return
        with self._lock:
            payload = b"".join(
                canonical_json_bytes(entry.model_dump(mode="json")) + b"\n"
                for entry in self._recorded_entries
            )
            _atomic_write(self.output_path, payload)
            digest = sha256_digest(payload)
            _atomic_write(
                self.output_path.with_name(f"{self.output_path.name}.sha256"),
                f"{digest}\n".encode(),
            )

    @property
    def recorded_entries(self) -> tuple[CassetteEntry, ...]:
        return tuple(self._recorded_entries)
