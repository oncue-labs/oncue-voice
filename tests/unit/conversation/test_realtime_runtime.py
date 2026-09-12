from collections.abc import AsyncIterator

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.conversation.realtime_runtime import RealtimeRuntime
from oncue_voice.providers.realtime.fake_provider import FakeRealtimeProvider
from oncue_voice.providers.realtime.models import (
    RealtimeEvent,
    RealtimeSessionOptions,
)


async def audio_chunks() -> AsyncIterator[bytes]:
    yield b"first"
    yield b"second"


def create_policy() -> DialoguePolicy:
    return DialoguePolicy(
        role="friendly helper",
        stages=("greeting",),
        goal="help the child get ready for bed",
        language="ko-KR",
        voice_id="alloy",
        scenario_context="The child is getting ready for bed.",
    )


async def test_realtime_runtime_forwards_audio_and_yields_provider_events() -> None:
    provider = FakeRealtimeProvider(
        (
            RealtimeEvent(type="audio_delta", audio=b"reply"),
            RealtimeEvent(type="response_completed"),
        )
    )
    runtime = RealtimeRuntime(provider)
    options = RealtimeSessionOptions(model="fake-realtime", voice_id="fake-voice")

    events = [
        event
        async for event in runtime.run(
            "session-1",
            create_policy(),
            audio_chunks(),
            options,
        )
    ]

    assert events == [
        RealtimeEvent(type="audio_delta", audio=b"reply"),
        RealtimeEvent(type="response_completed"),
    ]
    assert provider.session.received_audio == [b"first", b"second"]
    assert provider.session.close_count == 1


async def test_realtime_runtime_closes_session_when_provider_events_fail() -> None:
    provider = FakeRealtimeProvider(
        (RealtimeEvent(type="error", message="provider failed"),)
    )
    runtime = RealtimeRuntime(provider)
    options = RealtimeSessionOptions(model="fake-realtime", voice_id="fake-voice")

    events = [
        event
        async for event in runtime.run(
            "session-1",
            create_policy(),
            audio_chunks(),
            options,
        )
    ]

    assert events == [RealtimeEvent(type="error", message="provider failed")]
    assert provider.session.close_count == 1
