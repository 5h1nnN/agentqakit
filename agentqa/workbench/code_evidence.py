"""Workspace-confined browsing and explicit bounded code evidence selection."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from agentqa.errors import AgentQAError
from agentqa.models import CodeEvidence
from agentqa.specs.hashing import sha256_digest
from agentqa.storage.redaction import redact_text_secrets
from agentqa.workbench.models import WorkbenchProject, WorkspaceEntry

_MAX_FILE_BYTES = 256 * 1024
_MAX_LINES = 400
_MAX_ENTRIES = 500
_EXCLUDED_DIRECTORIES = {
    ".git",
    ".agentqa",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}
_SECRET_NAMES = {
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}
_SECRET_SUFFIXES = {".key", ".p12", ".pfx", ".pem"}
_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".kt",
    ".md",
    ".mjs",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


class CodeEvidenceError(AgentQAError):
    def __init__(self, message: str, *, code: str = "CODE_EVIDENCE_ERROR") -> None:
        super().__init__(message, code=code)


def _excluded_name(path: Path) -> bool:
    lower = path.name.lower()
    return (
        lower.startswith(".env")
        or lower in _SECRET_NAMES
        or path.suffix.lower() in _SECRET_SUFFIXES
    )


class CodeEvidenceService:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def project_root(self, project: WorkbenchProject) -> Path:
        relative = PurePosixPath(project.workspace_path)
        root = (self.workspace_root / Path(*relative.parts)).resolve()
        try:
            root.relative_to(self.workspace_root)
        except ValueError:
            raise CodeEvidenceError(
                "project workspace escapes the fixed workspace", code="CODE_EVIDENCE_PATH_ERROR"
            ) from None
        if not root.is_dir():
            raise CodeEvidenceError(
                "project workspace does not exist", code="CODE_EVIDENCE_PATH_MISSING"
            )
        return root

    def resolve(self, project: WorkbenchProject, value: str, *, file_only: bool) -> Path:
        if not value or "\x00" in value or "\\" in value:
            raise CodeEvidenceError("code path is invalid", code="CODE_EVIDENCE_PATH_ERROR")
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise CodeEvidenceError(
                "code path must be project-relative and cannot contain '..'",
                code="CODE_EVIDENCE_PATH_ERROR",
            )
        root = self.project_root(project)
        unresolved = root.joinpath(*relative.parts)
        cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise CodeEvidenceError(
                    "symlink paths cannot be used as code evidence",
                    code="CODE_EVIDENCE_PATH_ERROR",
                )
        resolved = unresolved.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise CodeEvidenceError(
                "code path escapes the project workspace", code="CODE_EVIDENCE_PATH_ERROR"
            ) from None
        if not resolved.exists():
            raise CodeEvidenceError("code path does not exist", code="CODE_EVIDENCE_PATH_MISSING")
        if any(part.lower() in _EXCLUDED_DIRECTORIES for part in relative.parts) or _excluded_name(
            resolved
        ):
            raise CodeEvidenceError(
                "sensitive or generated paths cannot be used as code evidence",
                code="CODE_EVIDENCE_PATH_BLOCKED",
            )
        if file_only and not resolved.is_file():
            raise CodeEvidenceError(
                "code evidence path must be a file", code="CODE_EVIDENCE_PATH_ERROR"
            )
        return resolved

    def browse(self, project: WorkbenchProject, path: str = ".") -> list[WorkspaceEntry]:
        directory = self.resolve(project, path, file_only=False)
        if not directory.is_dir():
            raise CodeEvidenceError(
                "browse path must be a directory", code="CODE_EVIDENCE_PATH_ERROR"
            )
        root = self.project_root(project)
        entries: list[WorkspaceEntry] = []
        for child in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name)):
            if len(entries) >= _MAX_ENTRIES:
                break
            if (
                child.is_symlink()
                or child.name.lower() in _EXCLUDED_DIRECTORIES
                or _excluded_name(child)
            ):
                continue
            if child.is_file() and child.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            entries.append(
                WorkspaceEntry(
                    path=child.relative_to(root).as_posix(),
                    name=child.name,
                    kind="directory" if child.is_dir() else "file",
                    size_bytes=None if child.is_dir() else child.stat().st_size,
                )
            )
        return entries

    def preview(
        self,
        project: WorkbenchProject,
        *,
        path: str,
        start_line: int,
        end_line: int,
        reason: str,
    ) -> CodeEvidence:
        if start_line < 1 or end_line < start_line or end_line - start_line + 1 > _MAX_LINES:
            raise CodeEvidenceError(
                "code selection must contain 1 to 400 ordered lines",
                code="CODE_EVIDENCE_LINE_LIMIT",
            )
        source = self.resolve(project, path, file_only=True)
        if source.suffix.lower() not in _TEXT_SUFFIXES:
            raise CodeEvidenceError(
                "file type is not allowed for code evidence", code="CODE_EVIDENCE_TYPE_ERROR"
            )
        size = source.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise CodeEvidenceError(
                "code file exceeds the 256 KiB preview limit",
                code="CODE_EVIDENCE_SIZE_ERROR",
            )
        try:
            raw = source.read_bytes()
            if b"\x00" in raw:
                raise UnicodeError
            lines = raw.decode("utf-8").splitlines()
        except (OSError, UnicodeError):
            raise CodeEvidenceError(
                "code file is not readable UTF-8 text", code="CODE_EVIDENCE_TYPE_ERROR"
            ) from None
        if start_line > len(lines) or end_line > len(lines):
            raise CodeEvidenceError(
                "code selection exceeds the file line count",
                code="CODE_EVIDENCE_LINE_LIMIT",
            )
        content = redact_text_secrets("\n".join(lines[start_line - 1 : end_line]))
        if len(content.encode("utf-8")) > 16_384:
            raise CodeEvidenceError(
                "code selection exceeds the 16384-byte excerpt limit",
                code="CODE_EVIDENCE_SIZE_ERROR",
            )
        relative = source.relative_to(self.workspace_root).as_posix()
        return CodeEvidence.model_validate(
            {
                "path": relative,
                "startLine": start_line,
                "endLine": end_line,
                "contentHash": sha256_digest(content.encode("utf-8")),
                "reason": reason,
                "content": content,
            }
        )


__all__ = ["CodeEvidenceError", "CodeEvidenceService"]
