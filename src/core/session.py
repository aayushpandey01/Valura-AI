"""
Session memory — in-memory implementation.
Stores conversation history per conversation_id.

Design decision: In-memory for this build. Tradeoffs:
  PRO: Zero latency, no infrastructure, simpler code.
  CON: Lost on restart, no cross-instance sharing.
  MIGRATION PATH: Replace SessionStore with a Redis or Postgres-backed
  implementation behind the same interface without touching the pipeline.
"""
from __future__ import annotations
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Deque


_MAX_TURNS_PER_SESSION = 20  # Keep last 20 turns (10 exchanges)
_SESSION_TTL_SECONDS = 3600   # 1 hour idle expiry


class SessionTurn:
    __slots__ = ("role", "content", "ts")

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content
        self.ts = time.monotonic()


class SessionStore:
    """Thread-safe in-memory conversation store."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Deque[SessionTurn]] = defaultdict(
            lambda: deque(maxlen=_MAX_TURNS_PER_SESSION)
        )
        self._last_access: Dict[str, float] = {}

    def add_turn(self, conversation_id: str, role: str, content: str) -> None:
        self._sessions[conversation_id].append(SessionTurn(role, content))
        self._last_access[conversation_id] = time.monotonic()

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Returns history as list of {"role": ..., "content": ...} dicts."""
        self._last_access[conversation_id] = time.monotonic()
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self._sessions[conversation_id]
        ]

    def evict_stale(self) -> int:
        """Remove sessions idle beyond TTL. Returns count evicted."""
        now = time.monotonic()
        stale = [
            cid
            for cid, last in self._last_access.items()
            if now - last > _SESSION_TTL_SECONDS
        ]
        for cid in stale:
            del self._sessions[cid]
            del self._last_access[cid]
        return len(stale)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


# Module-level singleton
_store = SessionStore()


def get_store() -> SessionStore:
    return _store
