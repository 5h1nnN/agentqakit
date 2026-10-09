"""Task-local record/replay session state."""

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import RLock
from typing import Literal

from agentqa.models import ReplayStats, RunContext
from agentqa.replay.store import JsonlCassetteStore


@dataclass(slots=True)
class ReplaySession:
    """Counters and storage isolated to one trial."""

    run_context: RunContext
    mode: Literal["off", "record", "replay"]
    store: JsonlCassetteStore
    allow_live_fallback: bool = False
    volatile_fields: tuple[str, ...] = ()
    counters: dict[tuple[str, str], int] = field(default_factory=dict)
    recorded: int = 0
    hits: int = 0
    misses: int = 0
    live_fallbacks: int = 0
    _lock: RLock = field(default_factory=RLock, repr=False)

    def next_call_index(self, kind: str, name: str) -> int:
        with self._lock:
            key = (kind, name)
            value = self.counters.get(key, 0) + 1
            self.counters[key] = value
            return value

    def note_recorded(self) -> None:
        with self._lock:
            self.recorded += 1

    def note_hit(self) -> None:
        with self._lock:
            self.hits += 1

    def note_miss(self) -> None:
        with self._lock:
            self.misses += 1

    def note_live_fallback(self) -> None:
        with self._lock:
            self.live_fallbacks += 1

    def stats(self) -> ReplayStats:
        with self._lock:
            lookups = self.hits + self.misses
            return ReplayStats(
                mode=self.mode,
                recorded=self.recorded,
                hits=self.hits,
                misses=self.misses,
                live_fallbacks=self.live_fallbacks,
                hit_rate=None if lookups == 0 else self.hits / lookups,
            )


_CURRENT_REPLAY_SESSION: ContextVar[ReplaySession | None] = ContextVar(
    "agentqa_current_replay_session",
    default=None,
)


def get_current_replay_session() -> ReplaySession | None:
    return _CURRENT_REPLAY_SESSION.get()


def set_current_replay_session(session: ReplaySession) -> Token[ReplaySession | None]:
    return _CURRENT_REPLAY_SESSION.set(session)


def reset_current_replay_session(token: Token[ReplaySession | None]) -> None:
    _CURRENT_REPLAY_SESSION.reset(token)
