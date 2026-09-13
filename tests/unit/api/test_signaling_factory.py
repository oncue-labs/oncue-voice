import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from oncue_voice.api.signaling import (
    DefaultSignalingSessionFactory,
    ManagedSignalingSession,
    UnknownCallSessionError,
)
from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.media.webrtc import SdpOffer, WebRtcSession
from oncue_voice.providers.realtime.models import RealtimeEvent
from oncue_voice.session.auth import VOICE_CONNECT_SCOPE
from oncue_voice.session.lifecycle import CallTerminationReason
from oncue_voice.session.models import ConnectionClaims, CreateSessionRequest
from oncue_voice.session.results import CallOutcome, CallStatus
from oncue_voice.session.service import SessionService
from oncue_voice.session.store import InMemoryVoiceSessionStore


def create_policy() -> DialoguePolicy:
    return DialoguePolicy(
        role="Santa",
        stages=("greeting",),
        goal="help the child get ready for bed",
        language="ko-KR",
        voiceId="alloy",
        scenarioContext="The child is preparing for bed.",
    )


def create_claims(*, user_id: int = 7) -> ConnectionClaims:
    now = int(datetime.now(timezone.utc).timestamp())
    return ConnectionClaims(
        callSessionId=1,
        userId=user_id,
        scope=(VOICE_CONNECT_SCOPE,),
        jti="token-1",
        iat=now,
        exp=now + 60,
    )


class FakePeerConnection:
    def __init__(self) -> None:
        self.added_tracks = []
        self.handlers = {}
        self.closed = False

    def addTrack(self, track) -> None:
        self.added_tracks.append(track)

    def on(self, event, handler):
        self.handlers[event] = handler
        return handler

    async def setRemoteDescription(self, description) -> None:
        return None

    async def createAnswer(self):
        return SimpleNamespace(sdp="answer-sdp")

    async def setLocalDescription(self, description) -> None:
        self.localDescription = description

    @property
    def localDescription(self):
        return getattr(self, "_local_description", None)

    @localDescription.setter
    def localDescription(self, value) -> None:
        self._local_description = value

    async def addIceCandidate(self, candidate) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class FakeResultCallback:
    def __init__(self) -> None:
        self.calls = []

    def send_result(self, call_session_id, result) -> None:
        self.calls.append((call_session_id, result))


def create_prepared_session(store: InMemoryVoiceSessionStore) -> None:
    SessionService(store).create(
        CreateSessionRequest(
            callSessionId=1,
            userId=7,
            policySnapshot=create_policy(),
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
    )


@pytest.mark.anyio
async def test_factory_builds_media_session_from_stored_policy_and_runtime() -> None:
    store = InMemoryVoiceSessionStore()
    create_prepared_session(store)
    peer = FakePeerConnection()
    runtime_sessions = []

    def runtime_runner_factory(session):
        runtime_sessions.append(session)

        async def runner(audio):
            async for _ in audio:
                pass
            if False:
                yield b""

        return runner

    factory = DefaultSignalingSessionFactory(
        store,
        runtime_runner_factory,
        peer_connection_factory=lambda: peer,
    )

    session = factory.create(1, create_claims())

    assert isinstance(session, ManagedSignalingSession)
    assert runtime_sessions[0].policy_snapshot == create_policy()
    assert peer.added_tracks

    await session.close()


@pytest.mark.anyio
async def test_factory_uses_coturn_environment_for_default_peer_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONCUE_VOICE_TURN_URLS", "turn:coturn:3478")
    monkeypatch.setenv("ONCUE_VOICE_TURN_USERNAME", "temporary-user")
    monkeypatch.setenv("ONCUE_VOICE_TURN_CREDENTIAL", "temporary-credential")
    store = InMemoryVoiceSessionStore()
    create_prepared_session(store)
    factory = DefaultSignalingSessionFactory(store, lambda session: _empty_runtime())

    session = factory.create(1, create_claims())
    peer = session._media._peer_connection
    configuration = peer.__dict__["_RTCPeerConnection__configuration"]
    ice_server = configuration.iceServers[0]

    try:
        assert ice_server.urls == ["turn:coturn:3478"]
        assert ice_server.username == "temporary-user"
        assert ice_server.credential == "temporary-credential"
    finally:
        await session.close()


@pytest.mark.anyio
async def test_factory_session_reports_user_hangup_result() -> None:
    store = InMemoryVoiceSessionStore()
    create_prepared_session(store)
    peer = FakePeerConnection()
    callback = FakeResultCallback()
    factory = DefaultSignalingSessionFactory(
        store,
        lambda session: _empty_runtime(),
        peer_connection_factory=lambda: peer,
        result_callback=callback,
    )

    session = factory.create(1, create_claims())
    await session.accept_offer(SdpOffer(sdp="offer"))
    result = await session.finish(CallTerminationReason.USER_HANGUP)

    assert result is not None
    assert result.call_status is CallStatus.IN_CALL
    assert result.call_outcome is CallOutcome.SUCCEEDED
    assert callback.calls[0][0] == 1
    await session.close()


@pytest.mark.anyio
async def test_factory_reports_provider_error_and_closes_media() -> None:
    store = InMemoryVoiceSessionStore()
    create_prepared_session(store)
    peer = FakePeerConnection()
    callback = FakeResultCallback()

    async def provider_error_runtime(audio):
        yield RealtimeEvent(type="error", message="provider failed")

    factory = DefaultSignalingSessionFactory(
        store,
        lambda session: provider_error_runtime,
        peer_connection_factory=lambda: peer,
        result_callback=callback,
    )
    session = factory.create(1, create_claims())

    await session.accept_offer(SdpOffer(sdp="offer"))
    await asyncio.sleep(0.01)

    assert callback.calls[0][1].call_outcome is CallOutcome.FAILED
    assert peer.closed is True
    await session.close()


@pytest.mark.anyio
async def test_factory_reports_time_limit_and_closes_media() -> None:
    store = InMemoryVoiceSessionStore()
    create_prepared_session(store)
    peer = FakePeerConnection()
    callback = FakeResultCallback()
    factory = DefaultSignalingSessionFactory(
        store,
        lambda session: _empty_runtime(),
        peer_connection_factory=lambda: peer,
        result_callback=callback,
        max_duration_seconds=0.0,
    )
    session = factory.create(1, create_claims())

    await session.accept_offer(SdpOffer(sdp="offer"))
    await asyncio.sleep(0.01)

    assert callback.calls[0][1].call_outcome is CallOutcome.FAILED
    assert peer.closed is True
    await session.close()


def test_factory_rejects_unknown_or_wrong_user_session() -> None:
    store = InMemoryVoiceSessionStore()
    create_prepared_session(store)
    factory = DefaultSignalingSessionFactory(
        store,
        lambda session: _empty_runtime(),
    )

    with pytest.raises(UnknownCallSessionError):
        factory.create(999, create_claims())
    with pytest.raises(UnknownCallSessionError):
        factory.create(1, create_claims(user_id=8))


def _empty_runtime():
    async def runner(audio):
        async for _ in audio:
            pass
        if False:
            yield b""

    return runner
