from collections.abc import AsyncIterator

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.providers.realtime.fake_provider import FakeRealtimeProvider
from oncue_voice.providers.realtime.models import (
    RealtimeEvent,
    RealtimeSessionOptions,
)


def create_policy() -> DialoguePolicy:
    return DialoguePolicy(
        role="Santa",
        stages=("greeting", "goal"),
        goal="help the child get ready for bed",
        language="ko-KR",
        voice_id="alloy",
        scenario_context="The child is getting ready for bed.",
    )

async def test_fake_realtime_provider_preserves_audio_and_returns_events() -> None:
    expected_events = (
        RealtimeEvent(type="speech_started"),
        RealtimeEvent(type="audio_delta", audio=b"response-audio"),
        RealtimeEvent(type="response_completed"),
    )
    provider = FakeRealtimeProvider(expected_events)
    options = RealtimeSessionOptions(model="fake-realtime", voice_id="fake-voice")

    session = await provider.connect(create_policy(), options)
    await session.send_audio_chunk(b"audio-part-1")
    await session.send_audio_chunk(b"audio-part-2")
    events = [event async for event in session.events()]

    assert provider.policy == create_policy()
    assert provider.options == options
    assert provider.session.received_audio == [b"audio-part-1", b"audio-part-2"]
    assert events == list(expected_events)


async def test_fake_realtime_session_close_is_idempotent() -> None:
    provider = FakeRealtimeProvider(())
    session = await provider.connect(
        create_policy(),
        RealtimeSessionOptions(model="fake-realtime", voice_id="fake-voice"),
    )

    await session.close()
    await session.close()

    assert provider.session.close_count == 1
