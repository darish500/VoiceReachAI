"""
app/services/conversation_store.py

In-memory implementation of the conversation persistence layer.

The Orchestrator depends only on the method shape used here
(`get_or_create`, `save`) -- see app/services/orchestrator.py's
`handle_message`, step 1 (`self.conversation_store.get_or_create(session_id)`)
and step 10 (`self.conversation_store.save(state)`). This class also
exposes a few extra management methods (load, delete, clear, list) that
are useful for the API layer and for tests/ops, without changing the
contract the Orchestrator relies on.

This store is intentionally process-local and non-persistent -- it is an
MVP substitute for a real backing store (Redis, Postgres, etc.). Because
FastAPI dependency injection provides a single shared instance per
process (see app/api/dependencies.py), all sessions live for the
lifetime of the running process.
"""

from __future__ import annotations

import logging
import threading

from app.core.models import ConversationState

logger = logging.getLogger(__name__)


class ConversationStore:
    """Thread-safe in-memory store of ConversationState, keyed by session_id."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str | None = None) -> ConversationState:
        """Create and persist a brand-new ConversationState.

        Args:
            session_id: Optional explicit session id. When omitted,
                ConversationState generates a fresh UUID for itself.

        Returns:
            The newly created ConversationState.
        """
        state = ConversationState(session_id=session_id) if session_id else ConversationState()
        with self._lock:
            self._sessions[state.session_id] = state
        logger.info("Created new conversation session_id=%s", state.session_id)
        return state

    def load(self, session_id: str) -> ConversationState | None:
        """Return the ConversationState for `session_id`, or None if absent."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> ConversationState:
        """Load an existing session or create a new one with this id.

        This is the exact method the Orchestrator calls at the start of
        every `handle_message` invocation.
        """
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing
            state = ConversationState(session_id=session_id)
            self._sessions[session_id] = state
        logger.info("Created new conversation session_id=%s", session_id)
        return state

    def save(self, state: ConversationState) -> None:
        """Persist (upsert) the given ConversationState, touching updated_at."""
        state.touch()
        with self._lock:
            self._sessions[state.session_id] = state
        logger.debug("Saved conversation session_id=%s", state.session_id)

    def update_history(self, session_id: str, state: ConversationState) -> None:
        """Explicitly persist an updated state for `session_id`.

        Provided as a semantically named alias of `save` for callers that
        want to be explicit that they are updating conversation history
        rather than creating a session for the first time.
        """
        if state.session_id != session_id:
            raise ValueError(
                f"state.session_id ({state.session_id}) does not match "
                f"the provided session_id ({session_id})"
            )
        self.save(state)

    def delete(self, session_id: str) -> bool:
        """Remove a session from the store.

        Returns:
            True if a session was found and removed, False otherwise.
        """
        with self._lock:
            removed = self._sessions.pop(session_id, None) is not None
        if removed:
            logger.info("Deleted conversation session_id=%s", session_id)
        return removed

    def clear(self) -> None:
        """Remove every session from the store. Primarily for tests/ops."""
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
        logger.info("Cleared conversation store (%d sessions removed)", count)

    def list_session_ids(self) -> list[str]:
        """Return all currently known session ids."""
        with self._lock:
            return list(self._sessions.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
