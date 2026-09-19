import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator, Callable
from fractions import Fraction

from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler

from oncue_voice.providers.realtime.models import RealtimeEvent


logger = logging.getLogger(__name__)


RuntimeAudioEvent = bytes | RealtimeEvent
RuntimeAudioRunner = Callable[
    [AsyncIterator[bytes]],
    AsyncIterator[RuntimeAudioEvent],
]
RuntimeCallback = Callable[[], object]


class PcmAudioInput:
    """Convert incoming WebRTC audio frames into mono s16 provider PCM."""

    def __init__(
        self,
        *,
        sample_rate: int,
        call_session_id: int | None = None,
    ) -> None:
        self._resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )
        self._chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._closed = False
        self._call_session_id = call_session_id
        self._received_frame_count = 0

    async def push_frame(self, frame: AudioFrame) -> None:
        if self._closed:
            return
        self._received_frame_count += 1
        if self._received_frame_count == 1:
            logger.info(
                "audio.input_frame_received callSessionId=%s "
                "sampleRate=%s samples=%s",
                self._call_session_id,
                frame.sample_rate,
                frame.samples,
            )
        for resampled in self._resampler.resample(frame):
            await self._chunks.put(_packed_pcm(resampled))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resampled in self._resampler.resample(None):
            await self._chunks.put(_packed_pcm(resampled))
        await self._chunks.put(None)

    async def chunks(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._chunks.get()
            if chunk is None:
                return
            yield chunk


def _packed_pcm(frame: AudioFrame) -> bytes:
    """Drop PyAV plane padding and return only packed mono s16 samples."""
    return bytes(frame.planes[0])[: frame.samples * 2]


class PcmAudioOutputTrack(MediaStreamTrack):
    """Expose provider mono s16 PCM as a paced aiortc audio track."""

    kind = "audio"
    _FRAME_DURATION_MS = 20

    def __init__(self, *, sample_rate: int) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._frame_samples = sample_rate * self._FRAME_DURATION_MS // 1000
        if self._frame_samples <= 0:
            raise ValueError("sample_rate must produce a positive 20ms frame")
        self._frame_bytes = self._frame_samples * 2
        self._chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._pending_pcm = bytearray()
        self._closed = False
        self._next_pts = 0
        self._started_at: float | None = None

    async def push_pcm(self, pcm: bytes) -> None:
        if self._closed:
            return
        if len(pcm) % 2:
            raise ValueError("mono s16 PCM must contain whole samples")
        self._pending_pcm.extend(pcm)
        while len(self._pending_pcm) >= self._frame_bytes:
            await self._chunks.put(bytes(self._pending_pcm[: self._frame_bytes]))
            del self._pending_pcm[: self._frame_bytes]

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending_pcm:
            self._pending_pcm.extend(
                b"\x00" * (self._frame_bytes - len(self._pending_pcm))
            )
            await self._chunks.put(bytes(self._pending_pcm))
            self._pending_pcm.clear()
        await self._chunks.put(None)
        super().stop()

    async def interrupt(self) -> None:
        """Discard provider audio that has not reached the WebRTC sender."""
        if self._closed:
            return
        while True:
            try:
                self._chunks.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._pending_pcm.clear()
        self._started_at = time.monotonic() - (
            self._next_pts / self._sample_rate
        )

    async def recv(self) -> AudioFrame:
        pcm = await self._chunks.get()
        if pcm is None:
            raise MediaStreamError

        if self._started_at is None:
            self._started_at = time.monotonic()
        else:
            wait = (
                self._started_at
                + (self._next_pts / self._sample_rate)
                - time.monotonic()
            )
            if wait > 0:
                await asyncio.sleep(wait)

        frame = AudioFrame(
            format="s16",
            layout="mono",
            samples=len(pcm) // 2,
        )
        frame.sample_rate = self._sample_rate
        frame.pts = self._next_pts
        frame.time_base = Fraction(1, self._sample_rate)
        self._next_pts += frame.samples
        frame.planes[0].update(pcm)
        return frame


class AudioRuntimeBridge:
    """Connect one runtime's audio output to the WebRTC output track."""

    def __init__(
        self,
        runtime: RuntimeAudioRunner,
        audio_input: PcmAudioInput,
        audio_output: PcmAudioOutputTrack,
        *,
        on_complete: RuntimeCallback | None = None,
        on_provider_error: RuntimeCallback | None = None,
    ) -> None:
        self._runtime = runtime
        self._audio_input = audio_input
        self._audio_output = audio_output
        self._on_complete = on_complete
        self._on_provider_error = on_provider_error
        self._output_chunk_count = 0

    async def run(self) -> None:
        logger.info("audio.runtime_started")
        try:
            async for event in self._runtime(self._audio_input.chunks()):
                logger.info(
                    "audio.runtime_event type=%s",
                    event.type if isinstance(event, RealtimeEvent) else "audio_bytes",
                )
                if isinstance(event, RealtimeEvent) and event.type == "error":
                    await self._notify(self._on_provider_error)
                    return
                if (
                    isinstance(event, RealtimeEvent)
                    and event.type == "speech_started"
                ):
                    await self._audio_output.interrupt()
                    continue
                pcm = self._audio_bytes(event)
                if pcm:
                    self._output_chunk_count += 1
                    if self._output_chunk_count == 1:
                        logger.info(
                            "audio.output_audio_received bytes=%s",
                            len(pcm),
                        )
                    await self._audio_output.push_pcm(pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("audio.runtime_failed")
            await self._notify(self._on_provider_error)
            raise
        else:
            logger.info(
                "audio.runtime_completed outputChunks=%s",
                self._output_chunk_count,
            )
            await self._notify(self._on_complete)
        finally:
            await self._audio_output.close()

    @staticmethod
    async def _notify(callback: RuntimeCallback | None) -> None:
        if callback is None:
            return
        result = callback()
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _audio_bytes(event: RuntimeAudioEvent) -> bytes | None:
        if isinstance(event, bytes):
            return event
        if event.type == "audio_delta":
            return event.audio
        return None
