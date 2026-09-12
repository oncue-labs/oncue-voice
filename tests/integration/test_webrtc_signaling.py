from datetime import datetime, timezone

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from oncue_voice.api.signaling import (
    UnknownCallSessionError,
    create_signaling_router,
)
from oncue_voice.media.webrtc import SdpAnswer
from oncue_voice.session.auth import ConnectionTokenVerifier
from oncue_voice.session.store import InMemoryJtiStore


def create_key_pair() -> tuple[str, str]:
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


def create_token(private_key: str, *, call_session_id: str = "call-1") -> str:
    now = int(datetime.now(timezone.utc).timestamp())
    return jwt.encode(
        {
            "callSessionId": call_session_id,
            "userId": "user-1",
            "scope": "voice:connect",
            "jti": f"token-{call_session_id}",
            "iat": now,
            "exp": now + 60,
        },
        private_key,
        algorithm="RS256",
    )


class FakeSignalingSession:
    def __init__(self) -> None:
        self.candidates = []
        self.offers = []
        self.closed = False

    async def accept_offer(self, offer):
        self.offers.append(offer)
        return SdpAnswer(sdp="answer-sdp")

    async def add_ice_candidate(self, candidate) -> None:
        self.candidates.append(candidate)

    async def close(self) -> None:
        self.closed = True


class FakeSignalingSessionFactory:
    def __init__(self, *, known: bool = True) -> None:
        self.session = FakeSignalingSession()
        self.known = known

    def create(self, call_session_id: str, claims) -> FakeSignalingSession:
        assert call_session_id == claims.call_session_id
        if not self.known:
            raise UnknownCallSessionError(call_session_id)
        return self.session


def create_client(
    *, known_session: bool = True
) -> tuple[TestClient, FakeSignalingSessionFactory, str]:
    private_key, public_key = create_key_pair()
    verifier = ConnectionTokenVerifier(public_key, InMemoryJtiStore())
    factory = FakeSignalingSessionFactory(known=known_session)
    app = FastAPI()
    app.include_router(create_signaling_router(verifier, factory))
    return TestClient(app), factory, create_token(private_key)


def test_signaling_accepts_offer_and_returns_symmetric_answer() -> None:
    client, factory, token = create_client()

    with client.websocket_connect(
        "/v1/signaling/call-sessions/call-1",
        headers={"Authorization": f"Bearer {token}"},
    ) as websocket:
        websocket.send_json({"type": "offer", "payload": {"sdp": "offer-sdp"}})

        assert websocket.receive_json() == {
            "type": "answer",
            "payload": {"sdp": "answer-sdp"},
        }
        assert factory.session.offers[0].sdp == "offer-sdp"


def test_signaling_forwards_ice_candidate_and_closes_on_hangup() -> None:
    client, factory, token = create_client()

    with client.websocket_connect(
        "/v1/signaling/call-sessions/call-1",
        headers={"Authorization": f"Bearer {token}"},
    ) as websocket:
        websocket.send_json(
            {
                "type": "ice-candidate",
                "payload": {
                    "candidate": "candidate:1 1 UDP 1 127.0.0.1 40000 typ host",
                    "sdpMid": "0",
                    "sdpMLineIndex": 0,
                },
            }
        )
        websocket.send_json({"type": "hangup"})

    assert factory.session.candidates[0].sdp_mid == "0"
    assert factory.session.closed is True


def test_signaling_rejects_missing_connection_token() -> None:
    client, _, _ = create_client()

    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect("/v1/signaling/call-sessions/call-1"):
            pass

    assert error.value.code == 1008


def test_signaling_rejects_unknown_call_session() -> None:
    client, _, token = create_client(known_session=False)

    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(
            "/v1/signaling/call-sessions/call-1",
            headers={"Authorization": f"Bearer {token}"},
        ):
            pass

    assert error.value.code == 1008


def test_signaling_rejects_invalid_sdp() -> None:
    client, _, token = create_client()

    with client.websocket_connect(
        "/v1/signaling/call-sessions/call-1",
        headers={"Authorization": f"Bearer {token}"},
    ) as websocket:
        websocket.send_json({"type": "offer", "payload": {"sdp": ""}})

        with pytest.raises(WebSocketDisconnect) as error:
            websocket.receive_json()

    assert error.value.code == 1003


def test_signaling_closes_session_after_abnormal_disconnect() -> None:
    client, factory, token = create_client()

    with client.websocket_connect(
        "/v1/signaling/call-sessions/call-1",
        headers={"Authorization": f"Bearer {token}"},
    ) as websocket:
        websocket.close(code=1001)

    assert factory.session.closed is True
