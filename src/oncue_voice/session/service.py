from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

from oncue_voice.session.models import (
    CreateSessionRequest,
    SessionStatus,
    VoiceSession,
)
from oncue_voice.session.store import VoiceSessionStore


class InvalidSessionError(ValueError):
    """Raised when a voice session request cannot be accepted."""


class SessionConflictError(InvalidSessionError):
    """Raised when a call session is reused with different data."""


class SessionNotFoundError(InvalidSessionError):
    """Raised when a voice session ID is unknown."""


class SessionService:
    def __init__(
        self,
        store: VoiceSessionStore,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def create(self, request: CreateSessionRequest) -> VoiceSession:
        now = self._utc(self._clock())
        expires_at = self._utc(request.expires_at)
        if expires_at <= now:
            raise InvalidSessionError("expires_at must be in the future")

        existing = self._store.get_by_call_session_id(request.call_session_id)
        if existing is not None and existing.status is not SessionStatus.CLOSED:
            if self._matches(existing, request):
                return existing
            raise SessionConflictError("call session already exists")

        session = VoiceSession(
            voiceSessionId=str(uuid4()),
            callSessionId=request.call_session_id,
            userId=request.user_id,
            policySnapshot=request.policy_snapshot,
            expiresAt=expires_at,
            status=SessionStatus.PREPARED,
            createdAt=now,
        )
        self._store.save(session)
        return session

    def get(self, voice_session_id: str) -> VoiceSession:
        session = self._store.get_by_voice_session_id(voice_session_id)
        if session is None:
            raise SessionNotFoundError("voice session not found")
        return session

    def close(self, voice_session_id: str, reason: str) -> None:
        _ = reason
        session = self.get(voice_session_id)
        if session.status is SessionStatus.CLOSED:
            return
        closed = session.model_copy(
            update={
                "status": SessionStatus.CLOSED,
                "closed_at": self._utc(self._clock()),
            }
        )
        self._store.save(closed)

    @staticmethod
    def _matches(session: VoiceSession, request: CreateSessionRequest) -> bool:
        return (
            session.user_id == request.user_id
            and session.policy_snapshot == request.policy_snapshot
            and session.expires_at == SessionService._utc(request.expires_at)
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
