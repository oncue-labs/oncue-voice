from typing import Protocol

from fastapi import APIRouter, WebSocket
from fastapi.websockets import WebSocketDisconnect
from pydantic import ValidationError

from oncue_voice.media.webrtc import IceCandidate, SdpOffer, WebRtcSession
from oncue_voice.session.auth import (
    ConnectionTokenVerifier,
    InvalidConnectionTokenError,
)
from oncue_voice.session.models import ConnectionClaims


class UnknownCallSessionError(LookupError):
    """Raised when a verified token refers to no prepared voice session."""


class SignalingSession(Protocol):
    async def accept_offer(self, offer: SdpOffer):
        """Accept an SDP offer and return an SDP answer."""

    async def add_ice_candidate(self, candidate: IceCandidate) -> None:
        """Apply a remote ICE candidate."""

    async def close(self) -> None:
        """Close the session's media resources."""


class SignalingSessionFactory(Protocol):
    def create(
        self,
        call_session_id: str,
        claims: ConnectionClaims,
    ) -> SignalingSession:
        """Create the WebRTC session for an authenticated call."""


class DefaultSignalingSessionFactory:
    def create(
        self,
        call_session_id: str,
        claims: ConnectionClaims,
    ) -> WebRtcSession:
        del call_session_id, claims
        return WebRtcSession()


def create_signaling_router(
    token_verifier: ConnectionTokenVerifier,
    session_factory: SignalingSessionFactory,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/v1/signaling/call-sessions/{call_session_id}")
    async def signaling_socket(
        websocket: WebSocket,
        call_session_id: str,
    ) -> None:
        token = _connection_token(websocket)
        if token is None:
            await websocket.close(code=1008)
            return

        try:
            claims = token_verifier.verify(token, call_session_id)
        except InvalidConnectionTokenError:
            await websocket.close(code=1008)
            return

        try:
            session = session_factory.create(call_session_id, claims)
        except UnknownCallSessionError:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            await _serve_messages(websocket, session)
        except WebSocketDisconnect:
            return
        finally:
            await session.close()

    return router


async def _serve_messages(
    websocket: WebSocket,
    session: SignalingSession,
) -> None:
    while True:
        message = await websocket.receive_json()
        if not isinstance(message, dict):
            await websocket.close(code=1003)
            return
        message_type = message.get("type")

        try:
            if message_type == "offer":
                offer = SdpOffer.model_validate(message.get("payload"))
                answer = await session.accept_offer(offer)
                await websocket.send_json(
                    {"type": "answer", "payload": answer.model_dump(mode="json")}
                )
                continue

            if message_type == "ice-candidate":
                candidate = IceCandidate.model_validate(message.get("payload"))
                await session.add_ice_candidate(candidate)
                continue
        except (ValidationError, ValueError):
            await websocket.close(code=1003)
            return

        if message_type == "hangup":
            return

        await websocket.close(code=1003)
        return


def _connection_token(websocket: WebSocket) -> str | None:
    authorization = websocket.headers.get("authorization")
    if authorization is None or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ")
    return token or None
