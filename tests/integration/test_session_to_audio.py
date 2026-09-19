import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
import pytest

from oncue_voice.api.signaling import (
    DefaultSignalingSessionFactory,
    create_signaling_router,
)
from oncue_voice.main import create_app
from oncue_voice.providers.realtime.models import RealtimeEvent
from oncue_voice.session.auth import ConnectionTokenVerifier
from oncue_voice.session.service import SessionService
from oncue_voice.session.store import InMemoryJtiStore, InMemoryVoiceSessionStore


class FakeRemoteAudioTrack:
    kind = "audio"

    def __init__(self, frame: AudioFrame) -> None:
        self._frame = frame
        self._sent = False

    async def recv(self) -> AudioFrame:
        if self._sent:
            raise MediaStreamError
        self._sent = True
        return self._frame


class FakePeerConnection:
    def __init__(self) -> None:
        self.added_tracks = []
        self.handlers = {}
        self.closed = False
        self.loop: asyncio.AbstractEventLoop | None = None

    def addTrack(self, track) -> None:
        self.added_tracks.append(track)

    def on(self, event, handler):
        self.loop = asyncio.get_running_loop()
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

    def emit_remote_track(self, track) -> None:
        assert self.loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._emit_remote_track(track),
            self.loop,
        )
        future.result(timeout=5)

    async def _emit_remote_track(self, track) -> None:
        task = self.handlers["track"](track)
        assert task is not None
        await task

    def receive_output(self) -> AudioFrame:
        assert self.loop is not None
        assert self.added_tracks
        future = asyncio.run_coroutine_threadsafe(
            self.added_tracks[0].recv(),
            self.loop,
        )
        return future.result(timeout=5)


class FakeResultCallback:
    def __init__(self) -> None:
        self.calls = []

    def send_result(self, call_session_id, result) -> None:
        self.calls.append((call_session_id, result))


class FakeSplitPipelineRuntime:
    def __init__(self, received_chunks: list[bytes]) -> None:
        self._received_chunks = received_chunks

    async def run(self, audio):
        self._received_chunks.append(await anext(audio))
        yield b"\x10\x00\x11\x00" * 240


class FakeRealtimeRuntime:
    def __init__(self, received_chunks: list[bytes]) -> None:
        self._received_chunks = received_chunks

    async def run(self, audio):
        self._received_chunks.append(await anext(audio))
        yield RealtimeEvent(
            type="audio_delta",
            audio=b"\x10\x00\x11\x00" * 240,
        )


def _key_pair() -> tuple[str, str]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


def _connection_token(private_key: str) -> str:
    now = int(datetime.now(timezone.utc).timestamp())
    return jwt.encode(
        {
            "callSessionId": 1,
            "userId": 7,
            "scope": "voice:connect",
            "jti": "jti-call-1",
            "iat": now,
            "exp": now + 60,
        },
        private_key,
        algorithm="RS256",
    )


def _audio_frame() -> AudioFrame:
    frame = AudioFrame(format="s16", layout="mono", samples=4)
    frame.sample_rate = 24_000
    frame.planes[0].update(b"\x01\x00\x02\x00\x03\x00\x04\x00")
    return frame


def _policy_payload() -> dict[str, object]:
    return {
        "role": "Coach",
        "stages": ["greeting"],
        "goal": "practice a short introduction",
        "language": "ko-KR",
        "voiceId": "alloy",
        "scenarioContext": "A short practice conversation.",
    }


@pytest.mark.parametrize("runtime_kind", ["split", "realtime"])
def test_session_creation_to_audio_runtime_and_final_callback(
    runtime_kind: str,
) -> None:
    private_key, public_key = _key_pair()
    store = InMemoryVoiceSessionStore()
    peer = FakePeerConnection()
    callback = FakeResultCallback()
    received_chunks: list[bytes] = []

    def runtime_runner_factory(session):
        assert session.policy_snapshot.role == "Coach"
        runtime = (
            FakeSplitPipelineRuntime(received_chunks)
            if runtime_kind == "split"
            else FakeRealtimeRuntime(received_chunks)
        )
        return runtime.run

    app = create_app(
        session_service=SessionService(store),
        internal_service_token="service-token",
    )
    app.include_router(
        create_signaling_router(
            ConnectionTokenVerifier(public_key, InMemoryJtiStore()),
            DefaultSignalingSessionFactory(
                store,
                runtime_runner_factory,
                peer_connection_factory=lambda: peer,
                result_callback=callback,
            ),
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/voice-sessions",
            headers={"Authorization": "Bearer service-token"},
            json={
                "callSessionId": 1,
                "userId": 7,
                "policySnapshot": _policy_payload(),
                "expiresAt": (
                    datetime.now(timezone.utc) + timedelta(minutes=5)
                ).isoformat(),
            },
        )
        assert response.status_code == 200
        created = response.json()
        assert created["callSessionId"] == 1
        assert created["voiceSessionId"]
        assert created["createdAt"]

        with client.websocket_connect(
            "/v1/signaling/call-sessions/1",
            headers={"Authorization": f"Bearer {_connection_token(private_key)}"},
        ) as websocket:
            websocket.send_json(
                {"type": "offer", "payload": {"sdp": "offer-sdp"}}
            )
            assert websocket.receive_json() == {
                "type": "answer",
                "payload": {"sdp": "answer-sdp"},
            }

            peer.emit_remote_track(FakeRemoteAudioTrack(_audio_frame()))
            output = peer.receive_output()
            websocket.send_json({"type": "hangup"})

    assert received_chunks
    assert received_chunks[0] == b"\x01\x00\x02\x00\x03\x00\x04\x00"
    assert output.samples == 480
    assert bytes(output.planes[0])[:4] == b"\x10\x00\x11\x00"
    assert peer.closed is True
    assert len(callback.calls) == 1
    call_session_id, result = callback.calls[0]
    assert call_session_id == 1
    assert result.voice_session_id == created["voiceSessionId"]
    assert result.call_status.value == "IN_CALL"
    assert result.call_outcome.value == "SUCCEEDED"

    persisted_session = store.get_by_call_session_id(1)
    assert persisted_session is not None
    assert set(persisted_session.model_dump()) == {
        "voice_session_id",
        "call_session_id",
        "user_id",
        "policy_snapshot",
        "expires_at",
        "status",
        "created_at",
        "closed_at",
    }
