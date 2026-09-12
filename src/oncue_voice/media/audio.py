import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from fractions import Fraction

from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler

from oncue_voice.providers.realtime.models import RealtimeEvent


RuntimeAudioEvent = bytes | RealtimeEvent
RuntimeAudioRunner = Callable[
    [AsyncIterator[bytes]],
    AsyncIterator[RuntimeAudioEvent],
]
RuntimeCallback = Callable[[], object]


class PcmAudioInput:
    """Convert incoming WebRTC audio frames into mono s16 provider PCM."""

    def __init__(self, *, sample_rate: int) -> None:
        self._resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )
        self._chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._closed = False

    async def push_frame(self, frame: AudioFrame) -> None:
        if self._closed:
            return
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
    """Expose provider mono s16 PCM as an aiortc audio track."""

    kind = "audio"

    def __init__(self, *, sample_rate: int) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._closed = False
        self._next_pts = 0

    async def push_pcm(self, pcm: bytes) -> None:
        if self._closed:
            return
        if len(pcm) % 2:
            raise ValueError("mono s16 PCM must contain whole samples")
        await self._chunks.put(pcm)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._chunks.put(None)
        super().stop()

    async def recv(self) -> AudioFrame:
        pcm = await self._chunks.get()
        if pcm is None:
            raise MediaStreamError

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

    async def run(self) -> None:
        try:
            async for event in self._runtime(self._audio_input.chunks()):
                if isinstance(event, RealtimeEvent) and event.type == "error":
                    await self._notify(self._on_provider_error)
                    return
                pcm = self._audio_bytes(event)
                if pcm:
                    await self._audio_output.push_pcm(pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._notify(self._on_provider_error)
            raise
        else:
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
