"""Deterministic campus service tools and their in-memory side-effect store."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def default_data_dir() -> Path:
    configured = os.environ.get("CAMPUS_AGENT_DATA_DIR")
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).resolve().parents[2] / "data",
        Path(sys.prefix) / "campus_ops_agent" / "data",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "courses.json").is_file():
            return candidate
    raise FileNotFoundError(
        "CampusOps data files were not found; set CAMPUS_AGENT_DATA_DIR to the data directory"
    )


def load_json(name: str, *, data_dir: Path | None = None) -> Any:
    path = (data_dir or default_data_dir()) / name
    return json.loads(path.read_text(encoding="utf-8"))


class CampusTools:
    """Five tool boundaries; registrations intentionally live for one service instance."""

    def __init__(self, *, data_dir: Path | None = None, enforce_idempotency: bool = True) -> None:
        self._courses: dict[str, JsonObject] = load_json("courses.json", data_dir=data_dir)
        self._profiles: dict[str, JsonObject] = load_json("profiles.json", data_dir=data_dir)
        self._scores: dict[str, JsonObject] = load_json("scores.json", data_dir=data_dir)
        self._events: dict[str, JsonObject] = load_json("events.json", data_dir=data_dir)
        self._enforce_idempotency = enforce_idempotency
        self._registrations: dict[str, JsonObject] = {}
        self._sequence = 0

    @property
    def events(self) -> dict[str, JsonObject]:
        return self._events

    def query_course(self, course_id: str, simulate_timeout: bool = False) -> JsonObject:
        if simulate_timeout:
            raise TimeoutError("synthetic course service timeout")
        value = self._courses.get(course_id)
        return dict(value) if value else {"status": "not_found"}

    def query_score(self, subject_user_id: str, course_id: str) -> JsonObject:
        value = self._scores.get(f"{subject_user_id}:{course_id}")
        if value is None:
            return {"status": "not_found"}
        return {"status": "ok", "userId": subject_user_id, **value}

    def register_event(self, user_id: str, event_id: str, idempotency_key: str) -> JsonObject:
        if event_id not in self._events:
            return {"status": "not_found"}
        if self._enforce_idempotency and idempotency_key in self._registrations:
            previous = self._registrations[idempotency_key]
            return {**previous, "status": "already_registered"}
        self._sequence += 1
        result: JsonObject = {
            "status": "registered",
            "registrationId": f"reg-{self._sequence:04d}",
            "userId": user_id,
            "eventId": event_id,
        }
        self._registrations[idempotency_key] = result
        return dict(result)

    def get_profile(self, subject_user_id: str) -> JsonObject:
        value = self._profiles.get(subject_user_id)
        return dict(value) if value else {"status": "not_found"}


def wrap_tools(tools: CampusTools, runtime: Any) -> dict[str, Callable[..., Any]]:
    """Apply AgentQAKit interception only when a runtime was explicitly supplied."""

    names = ("query_course", "query_score", "register_event", "get_profile")
    if runtime is None:
        return {name: getattr(tools, name) for name in names}
    return {name: runtime.wrap_tool(getattr(tools, name), name=name) for name in names}
