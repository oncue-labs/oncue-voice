import asyncio
from types import SimpleNamespace

import pytest
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from oncue_voice.media.audio import PcmAudioInput, PcmAudioOutputTrack
from oncue_voice.media.webrtc import IceCandidate, SdpOffer, WebRtcSession


class FakePeerConnection:
    def __init__(self) -> None:
        self.remote_description = None
        self.local_description = None
        self.received_candidates = []
        self.close_count = 0
        self.added_tracks = []
        self.track_handler = None

    async def setRemoteDescription(self, description) -> None:
        self.remote_description = description

    async def createAnswer(self):
        return SimpleNamespace(type="answer", sdp="answer-sdp")

    async def setLocalDescription(self, description) -> None:
        self.local_description = description

    @property
    def localDescription(self):
        return self.local_description

    async def addIceCandidate(self, candidate) -> None:
        self.received_candidates.append(candidate)

    async def close(self) -> None:
        self.close_count += 1

    def addTrack(self, track) -> None:
        self.added_tracks.append(track)

    def on(self, event, handler):
        assert event == "track"
        self.track_handler = handler
        return handler


@pytest.mark.anyio
async def test_accept_offer_returns_answer_with_contract_sdp_field() -> None:
    peer = FakePeerConnection()
    session = WebRtcSession(peer)

    answer = await session.accept_offer(SdpOffer(sdp="offer-sdp"))

    assert answer.model_dump(by_alias=True) == {"sdp": "answer-sdp"}
    assert peer.remote_description.sdp == "offer-sdp"
    assert peer.remote_description.type == "offer"
    assert peer.local_description.sdp == "answer-sdp"


@pytest.mark.anyio
async def test_add_ice_candidate_preserves_contract_fields() -> None:
    peer = FakePeerConnection()
    session = WebRtcSession(peer)
    candidate = IceCandidate(
        candidate="candidate:1 1 UDP 1 127.0.0.1 40000 typ host",
        sdp_mid="0",
        sdp_m_line_index=0,
    )

    await session.add_ice_candidate(candidate)

    received = peer.received_candidates[0]
    assert received.foundation == "candidate:1"
    assert received.ip == "127.0.0.1"
    assert received.port == 40000
    assert received.sdpMid == candidate.sdp_mid
    assert received.sdpMLineIndex == candidate.sdp_m_line_index


@pytest.mark.anyio
async def test_close_is_idempotent() -> None:
    peer = FakePeerConnection()
    session = WebRtcSession(peer)

    await session.close()
    await session.close()

    assert peer.close_count == 1


class FakeRemoteAudioTrack:
    kind = "audio"

    def __init__(self) -> None:
        self._sent = False

    async def recv(self) -> AudioFrame:
        if self._sent:
            raise MediaStreamError
        self._sent = True
        frame = AudioFrame(format="s16", layout="mono", samples=2)
        frame.sample_rate = 24_000
        frame.planes[0].update(b"\x07\x00" * 2)
        return frame


class FakeRuntimeBridge:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False
        self._release = asyncio.Event()

    async def run(self) -> None:
        self.started.set()
        try:
            await self._release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.anyio
async def test_media_tracks_are_bound_and_remote_audio_is_consumed() -> None:
    peer = FakePeerConnection()
    audio_input = PcmAudioInput(sample_rate=24_000)
    audio_output = PcmAudioOutputTrack(sample_rate=24_000)
    session = WebRtcSession(
        peer,
        audio_input=audio_input,
        audio_output=audio_output,
    )

    assert peer.added_tracks == [audio_output]
    assert peer.track_handler is not None

    track_task = peer.track_handler(FakeRemoteAudioTrack())
    assert isinstance(track_task, asyncio.Task)
    await track_task
    await session.close()

    assert [chunk async for chunk in audio_input.chunks()] == [b"\x07\x00" * 2]


@pytest.mark.anyio
async def test_accept_offer_starts_runtime_and_close_cancels_it() -> None:
    peer = FakePeerConnection()
    runtime_bridge = FakeRuntimeBridge()
    session = WebRtcSession(peer, runtime_bridge=runtime_bridge)

    await session.accept_offer(SdpOffer(sdp="offer-sdp"))
    await runtime_bridge.started.wait()

    await session.close()

    assert runtime_bridge.cancelled is True
