import asyncio

import pytest
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from oncue_voice.media.audio import (
    AudioRuntimeBridge,
    PcmAudioInput,
    PcmAudioOutputTrack,
)
from oncue_voice.providers.realtime.models import RealtimeEvent


def create_mono_pcm_frame(
    *,
    sample_rate: int,
    samples: int,
    value: int = 0,
) -> AudioFrame:
    frame = AudioFrame(format="s16", layout="mono", samples=samples)
    frame.sample_rate = sample_rate
    frame.planes[0].update(value.to_bytes(2, "little", signed=True) * samples)
    return frame


@pytest.mark.anyio
async def test_audio_input_resamples_webrtc_frame_to_provider_pcm() -> None:
    audio_input = PcmAudioInput(sample_rate=24_000)
    await audio_input.push_frame(
        create_mono_pcm_frame(sample_rate=48_000, samples=480, value=7)
    )
    await audio_input.close()

    chunks = [chunk async for chunk in audio_input.chunks()]

    pcm = b"".join(chunks)

    assert len(pcm) == 480
    assert pcm[:2] == (7).to_bytes(2, "little", signed=True)


@pytest.mark.anyio
async def test_audio_output_track_exposes_provider_pcm_as_audio_frame() -> None:
    audio_track = PcmAudioOutputTrack(sample_rate=24_000)
    await audio_track.push_pcm(b"\x07\x00" * 480)

    frame = await audio_track.recv()

    assert frame.format.name == "s16"
    assert frame.layout.name == "mono"
    assert frame.sample_rate == 24_000
    assert frame.samples == 480


@pytest.mark.anyio
async def test_audio_output_track_raises_stream_error_after_close() -> None:
    audio_track = PcmAudioOutputTrack(sample_rate=24_000)
    await audio_track.close()

    with pytest.raises(MediaStreamError):
        await audio_track.recv()


@pytest.mark.anyio
async def test_audio_runtime_bridge_forwards_split_and_realtime_audio() -> None:
    audio_input = PcmAudioInput(sample_rate=24_000)
    audio_output = PcmAudioOutputTrack(sample_rate=24_000)
    completed = []

    async def runtime(audio):
        received = [chunk async for chunk in audio]
        assert received == [b"\x07\x00" * 2]
        yield b"\x08\x00" * 2
        yield RealtimeEvent(type="audio_delta", audio=b"\x09\x00" * 2)

    bridge = AudioRuntimeBridge(
        runtime,
        audio_input,
        audio_output,
        on_complete=lambda: completed.append("completed"),
    )
    bridge_task = asyncio.create_task(bridge.run())
    await audio_input.push_frame(
        create_mono_pcm_frame(sample_rate=24_000, samples=2, value=7)
    )
    await audio_input.close()
    await bridge_task

    first_frame = await audio_output.recv()
    second_frame = await audio_output.recv()
    assert bytes(first_frame.planes[0])[:4] == b"\x08\x00" * 2
    assert bytes(second_frame.planes[0])[:4] == b"\x09\x00" * 2
    with pytest.raises(MediaStreamError):
        await audio_output.recv()
    assert completed == ["completed"]


@pytest.mark.anyio
async def test_audio_runtime_bridge_reports_realtime_provider_error() -> None:
    audio_input = PcmAudioInput(sample_rate=24_000)
    audio_output = PcmAudioOutputTrack(sample_rate=24_000)
    errors = []

    async def runtime(audio):
        yield RealtimeEvent(type="error", message="provider failed")
        if False:
            yield b""

    bridge = AudioRuntimeBridge(
        runtime,
        audio_input,
        audio_output,
        on_provider_error=lambda: errors.append("provider"),
    )

    await bridge.run()

    assert errors == ["provider"]
    with pytest.raises(MediaStreamError):
        await audio_output.recv()
