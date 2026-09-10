from datetime import datetime, timedelta, timezone

import pytest

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.session.service import (
    CreateSessionRequest,
    InvalidSessionError,
    SessionConflictError,
    SessionNotFoundError,
    SessionService,
    SessionStatus,
)
from oncue_voice.session.store import InMemoryVoiceSessionStore


def create_policy() -> DialoguePolicy:
    return DialoguePolicy(
        role="Santa",
        stages=("greeting", "goal"),
        goal="help the child get ready for bed",
        language="ko-KR",
        voice_id="alloy",
        scenario_context="The child is preparing for bed.",
    )


def create_request(
    call_session_id: str = "call-1",
    user_id: str = "user-1",
    expires_at: str | None = None,
) -> CreateSessionRequest:
    expiry = expires_at or (
        datetime.now(timezone.utc) + timedelta(minutes=5)
    ).isoformat()
    return CreateSessionRequest(
        call_session_id=call_session_id,
        user_id=user_id,
        policy=create_policy(),
        expires_at=expiry,
    )


def test_create_generates_voice_session_and_stores_policy_snapshot() -> None:
    store = InMemoryVoiceSessionStore()
    service = SessionService(store)

    session = service.create(create_request())

    assert session.voice_session_id
    assert session.call_session_id == "call-1"
    assert session.user_id == "user-1"
    assert session.status is SessionStatus.PREPARED
    assert session.policy.goal == "help the child get ready for bed"
    assert store.get_by_voice_session_id(session.voice_session_id) == session
    assert store.get_by_call_session_id("call-1") == session


def test_create_is_idempotent_for_same_call_session() -> None:
    service = SessionService(InMemoryVoiceSessionStore())
    request = create_request()

    first = service.create(request)
    second = service.create(request)

    assert second == first


def test_create_rejects_conflicting_duplicate_call_session() -> None:
    service = SessionService(InMemoryVoiceSessionStore())
    service.create(create_request())

    with pytest.raises(SessionConflictError, match="call session already exists"):
        service.create(create_request(user_id="user-2"))


def test_create_rejects_expired_session() -> None:
    service = SessionService(InMemoryVoiceSessionStore())
    expired_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()

    with pytest.raises(InvalidSessionError, match="expires_at must be in the future"):
        service.create(create_request(expires_at=expired_at))


def test_close_marks_session_closed_and_is_idempotent() -> None:
    service = SessionService(InMemoryVoiceSessionStore())
    session = service.create(create_request())

    service.close(session.voice_session_id, "user_hangup")
    service.close(session.voice_session_id, "duplicate_hangup")

    closed = service.get(session.voice_session_id)
    assert closed.status is SessionStatus.CLOSED
    assert closed.closed_at is not None


def test_close_rejects_unknown_voice_session() -> None:
    service = SessionService(InMemoryVoiceSessionStore())

    with pytest.raises(SessionNotFoundError, match="voice session not found"):
        service.close("unknown", "user_hangup")
