import asyncio
import logging
from typing import Any, Protocol

from collections.abc import Callable

from aiortc import (
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import candidate_from_sdp
from pydantic import BaseModel, ConfigDict, Field, field_validator

from oncue_voice.config import TurnSettings
from oncue_voice.media.audio import (
    AudioRuntimeBridge,
    PcmAudioInput,
    PcmAudioOutputTrack,
)


logger = logging.getLogger(__name__)


class SdpOffer(BaseModel):
    """SDP offer sent by the mobile WebRTC peer."""

    sdp: str

    @field_validator("sdp")
    @classmethod
    def validate_sdp(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sdp must not be empty")
        return value


class SdpAnswer(BaseModel):
    """SDP answer returned to the mobile WebRTC peer."""

    sdp: str


class IceCandidate(BaseModel):
    """ICE candidate exchanged by the WebRTC peers."""

    model_config = ConfigDict(populate_by_name=True)

    candidate: str
    sdp_mid: str | None = Field(alias="sdpMid")
    sdp_m_line_index: int | None = Field(alias="sdpMLineIndex")


class PeerConnection(Protocol):
    def addTrack(self, track: PcmAudioOutputTrack) -> Any:
        """Add the local audio track to the peer connection."""

    def on(self, event: str, handler: Any) -> Any:
        """Register an event handler on the peer connection."""

    async def setRemoteDescription(self, description: Any) -> None:
        """Set the remote SDP description."""

    async def createAnswer(self) -> Any:
        """Create an SDP answer."""

    async def setLocalDescription(self, description: Any) -> None:
        """Set the local SDP description."""

    async def addIceCandidate(self, candidate: RTCIceCandidate) -> None:
        """Add a remote ICE candidate."""

    async def close(self) -> None:
        """Close the peer connection."""


def create_peer_connection_factory(
    turn_settings: TurnSettings,
) -> Callable[[], RTCPeerConnection]:
    """Create peer connections using the service's coturn credentials."""

    def create_peer_connection() -> RTCPeerConnection:
        configuration = RTCConfiguration(
            iceServers=[
                RTCIceServer(
                    urls=list(turn_settings.urls),
                    username=turn_settings.username,
                    credential=turn_settings.credential,
                )
            ]
        )
        return RTCPeerConnection(configuration=configuration)

    return create_peer_connection


class WebRtcSession:
    """Owns one authenticated WebRTC peer connection."""

    def __init__(
        self,
        peer_connection: PeerConnection | None = None,
        *,
        audio_input: PcmAudioInput | None = None,
        audio_output: PcmAudioOutputTrack | None = None,
        runtime_bridge: AudioRuntimeBridge | None = None,
        call_session_id: int | None = None,
    ) -> None:
        self._peer_connection = peer_connection or RTCPeerConnection()
        self._call_session_id = call_session_id
        self._audio_input = audio_input
        self._audio_output = audio_output
        self._runtime_bridge = runtime_bridge
        self._runtime_task: asyncio.Task[None] | None = None
        self._track_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        if audio_output is not None:
            self._peer_connection.addTrack(audio_output)
        if audio_input is not None:
            self._peer_connection.on("track", self._on_track)
        self._peer_connection.on(
            "iceconnectionstatechange", self._on_ice_connection_state_change
        )
        self._peer_connection.on(
            "connectionstatechange", self._on_connection_state_change
        )

    async def accept_offer(self, offer: SdpOffer) -> SdpAnswer:
        if self._runtime_bridge is not None and self._runtime_task is None:
            self._runtime_task = asyncio.create_task(self._runtime_bridge.run())
        await self._peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=offer.sdp, type="offer")
        )
        answer = await self._peer_connection.createAnswer()
        await self._peer_connection.setLocalDescription(answer)
        local_description = self._peer_connection.localDescription
        answer_sdp = getattr(local_description, "sdp", None) or answer.sdp
        logger.info(
            "webrtc.offer_accepted callSessionId=%s", self._call_session_id
        )
        logger.info(
            "webrtc.answer_created callSessionId=%s", self._call_session_id
        )
        return SdpAnswer(sdp=answer_sdp)

    async def add_ice_candidate(self, candidate: IceCandidate) -> None:
        remote_candidate = candidate_from_sdp(candidate.candidate)
        remote_candidate.sdpMid = candidate.sdp_mid
        remote_candidate.sdpMLineIndex = candidate.sdp_m_line_index
        await self._peer_connection.addIceCandidate(remote_candidate)
        logger.info(
            "webrtc.ice_candidate_received callSessionId=%s",
            self._call_session_id,
        )

    def _on_ice_connection_state_change(self) -> None:
        logger.info(
            "webrtc.ice_connection_state callSessionId=%s state=%s",
            self._call_session_id,
            getattr(self._peer_connection, "iceConnectionState", "unknown"),
        )

    def _on_connection_state_change(self) -> None:
        logger.info(
            "webrtc.connection_state callSessionId=%s state=%s",
            self._call_session_id,
            getattr(self._peer_connection, "connectionState", "unknown"),
        )

    def _on_track(self, track: Any) -> asyncio.Task[None] | None:
        logger.info(
            "webrtc.remote_track_received callSessionId=%s kind=%s",
            self._call_session_id,
            getattr(track, "kind", "unknown"),
        )
        if self._audio_input is None or getattr(track, "kind", None) != "audio":
            return None
        task = asyncio.create_task(self._consume_audio_track(track))
        self._track_tasks.add(task)
        task.add_done_callback(self._track_tasks.discard)
        return task

    async def _consume_audio_track(self, track: Any) -> None:
        frame_count = 0
        try:
            while not self._closed:
                frame = await track.recv()
                frame_count += 1
                if frame_count == 1:
                    logger.info(
                        "webrtc.remote_audio_frame_received callSessionId=%s "
                        "sampleRate=%s samples=%s",
                        self._call_session_id,
                        frame.sample_rate,
                        frame.samples,
                    )
                await self._audio_input.push_frame(frame)
        except MediaStreamError:
            return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._audio_input is not None:
            await self._audio_input.close()
        current_task = asyncio.current_task()
        if self._runtime_task is not None and self._runtime_task is not current_task:
            if not self._runtime_task.done():
                self._runtime_task.cancel()
            try:
                await self._runtime_task
            except asyncio.CancelledError:
                pass
        for task in tuple(self._track_tasks):
            if not task.done():
                task.cancel()
        for task in tuple(self._track_tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._audio_output is not None:
            await self._audio_output.close()
        await self._peer_connection.close()
