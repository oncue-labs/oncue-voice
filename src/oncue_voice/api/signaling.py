import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol

from fastapi import APIRouter, WebSocket
from fastapi.websockets import WebSocketDisconnect
from pydantic import ValidationError

from oncue_voice.config import TurnSettings
from oncue_voice.media.audio import (
    AudioRuntimeBridge,
    PcmAudioInput,
    PcmAudioOutputTrack,
    RuntimeAudioRunner,
)
from oncue_voice.media.webrtc import (
    IceCandidate,
    PeerConnection,
    SdpAnswer,
    SdpOffer,
    WebRtcSession,
    create_peer_connection_factory,
)
from oncue_voice.session.auth import (
    ConnectionTokenVerifier,
    InvalidConnectionTokenError,
)
from oncue_voice.session.models import ConnectionClaims, SessionStatus, VoiceSession
from oncue_voice.session.lifecycle import (
    CallLifecycle,
    CallResultCallback,
    CallTerminationReason,
)
from oncue_voice.session.store import VoiceSessionStore


class UnknownCallSessionError(LookupError):
    """Raised when a verified token refers to no prepared voice session."""


class SignalingSession(Protocol):
    async def accept_offer(self, offer: SdpOffer) -> SdpAnswer:
        """Accept an SDP offer and return an SDP answer."""

    async def add_ice_candidate(self, candidate: IceCandidate) -> None:
        """Apply a remote ICE candidate."""

    async def close(self) -> None:
        """Close the session's media resources."""

    async def finish(self, reason: CallTerminationReason) -> None:
        """Report the internal termination reason once."""


class SignalingSessionFactory(Protocol):
    def create(
        self,
        call_session_id: str,
        claims: ConnectionClaims,
    ) -> SignalingSession:
        """Create the WebRTC session for an authenticated call."""


RuntimeRunnerFactory = Callable[[VoiceSession], RuntimeAudioRunner]


class DefaultSignalingSessionFactory:
    def __init__(
        self,
        voice_session_store: VoiceSessionStore,
        runtime_runner_factory: RuntimeRunnerFactory,
        *,
        peer_connection_factory: Callable[[], PeerConnection] | None = None,
        sample_rate: int = 24_000,
        clock: Callable[[], datetime] | None = None,
        result_callback: CallResultCallback | None = None,
        max_duration_seconds: float = 300.0,
    ) -> None:
        self._voice_session_store = voice_session_store
        self._runtime_runner_factory = runtime_runner_factory
        self._peer_connection_factory = peer_connection_factory
        self._sample_rate = sample_rate
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._result_callback = result_callback
        self._max_duration_seconds = max_duration_seconds

    def create(
        self,
        call_session_id: str,
        claims: ConnectionClaims,
    ) -> SignalingSession:
        session = self._voice_session_store.get_by_call_session_id(call_session_id)
        if not self._is_usable(session, call_session_id, claims):
            raise UnknownCallSessionError(call_session_id)

        audio_input = PcmAudioInput(sample_rate=self._sample_rate)
        audio_output = PcmAudioOutputTrack(sample_rate=self._sample_rate)
        lifecycle = CallLifecycle(
            call_session_id=session.call_session_id,
            voice_session_id=session.voice_session_id,
            result_callback=self._result_callback,
            clock=self._clock,
        )
        managed_session = ManagedSignalingSession(
            lifecycle,
            max_duration_seconds=self._max_duration_seconds,
        )
        runtime = self._runtime_runner_factory(session)
        bridge = AudioRuntimeBridge(
            runtime,
            audio_input,
            audio_output,
            on_complete=managed_session.scenario_completed,
            on_provider_error=managed_session.provider_error,
        )
        peer_connection_factory = self._peer_connection_factory
        if peer_connection_factory is None:
            peer_connection_factory = create_peer_connection_factory(
                TurnSettings.from_environment()
            )
        peer_connection = peer_connection_factory()
        managed_session.attach_media(
            WebRtcSession(
                peer_connection,
                audio_input=audio_input,
                audio_output=audio_output,
                runtime_bridge=bridge,
            )
        )
        return managed_session

    def _is_usable(
        self,
        session: VoiceSession | None,
        call_session_id: str,
        claims: ConnectionClaims,
    ) -> bool:
        if session is None:
            return False
        if session.call_session_id != call_session_id:
            return False
        if session.user_id != claims.user_id:
            return False
        if session.status is SessionStatus.CLOSED:
            return False
        expires_at = session.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at > self._clock().astimezone(timezone.utc)


class ManagedSignalingSession:
    """Bind WebRTC media, runtime termination, and final result reporting."""

    def __init__(
        self,
        lifecycle: CallLifecycle,
        *,
        max_duration_seconds: float,
    ) -> None:
        self._lifecycle = lifecycle
        self._max_duration_seconds = max_duration_seconds
        self._media: WebRtcSession | None = None
        self._time_limit_task: asyncio.Task[None] | None = None

    def attach_media(self, media: WebRtcSession) -> None:
        self._media = media

    async def accept_offer(self, offer: SdpOffer) -> SdpAnswer:
        if self._media is None:
            raise RuntimeError("media session is not attached")
        answer = await self._media.accept_offer(offer)
        self._lifecycle.mark_in_call()
        if self._time_limit_task is None:
            self._time_limit_task = asyncio.create_task(self._enforce_time_limit())
        return answer

    async def add_ice_candidate(self, candidate: IceCandidate) -> None:
        if self._media is None:
            raise RuntimeError("media session is not attached")
        await self._media.add_ice_candidate(candidate)

    async def finish(self, reason: CallTerminationReason):
        return await self._lifecycle.finish(reason)

    async def close(self) -> None:
        current_task = asyncio.current_task()
        if (
            self._time_limit_task is not None
            and self._time_limit_task is not current_task
            and not self._time_limit_task.done()
        ):
            self._time_limit_task.cancel()
            try:
                await self._time_limit_task
            except asyncio.CancelledError:
                pass
        if self._media is not None:
            await self._media.close()

    async def scenario_completed(self) -> None:
        await self.finish(CallTerminationReason.SCENARIO_COMPLETED)
        await self.close()

    async def provider_error(self) -> None:
        await self.finish(CallTerminationReason.PROVIDER_ERROR)
        await self.close()

    async def _enforce_time_limit(self) -> None:
        await asyncio.sleep(self._max_duration_seconds)
        await self.finish(CallTerminationReason.TIME_LIMIT)
        await self.close()


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
        termination_reason = CallTerminationReason.ABNORMAL_DISCONNECT
        try:
            termination_reason = await _serve_messages(websocket, session)
        except WebSocketDisconnect:
            return
        finally:
            finish = getattr(session, "finish", None)
            if finish is not None:
                await finish(termination_reason)
            await session.close()

    return router


async def _serve_messages(
    websocket: WebSocket,
    session: SignalingSession,
) -> CallTerminationReason:
    while True:
        message = await websocket.receive_json()
        if not isinstance(message, dict):
            await websocket.close(code=1003)
            return CallTerminationReason.ABNORMAL_DISCONNECT
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
            return CallTerminationReason.ABNORMAL_DISCONNECT

        if message_type == "hangup":
            return CallTerminationReason.USER_HANGUP

        await websocket.close(code=1003)
        return CallTerminationReason.ABNORMAL_DISCONNECT


def _connection_token(websocket: WebSocket) -> str | None:
    authorization = websocket.headers.get("authorization")
    if authorization is None or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ")
    return token or None
