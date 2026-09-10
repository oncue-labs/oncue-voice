import pytest

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.conversation.events import ConversationEvent
from oncue_voice.conversation.runtime import (
    ConversationRuntime,
    ConversationRuntimeOptions,
)
from oncue_voice.providers.fake_llm_provider import FakeLlmProvider
from oncue_voice.providers.fake_stt_provider import FakeSttProvider
from oncue_voice.providers.fake_tts_provider import FakeTtsProvider
from oncue_voice.providers.models import ProviderBundle


async def audio_chunks() -> bytes:
    yield b"input-audio"


def create_policy(forbidden_topics: tuple[str, ...] = ()) -> DialoguePolicy:
    return DialoguePolicy(
        role="friendly helper",
        stages=("greeting", "goal"),
        goal="help the child get ready for bed",
        allowed_topics=("bedtime",),
        forbidden_topics=forbidden_topics,
        termination_conditions=("goal reached",),
        language="ko-KR",
        voice_id="fake-voice",
        instructions=("Use a warm tone.",),
        dialogue_rules=("Ask one question at a time.",),
        scenario_context="The child is getting ready for bed.",
        voice_settings={"speed": 1.0},
    )


def create_bundle(
    events: list[str],
    transcript_text: str = "hello",
) -> ProviderBundle:
    stt = FakeSttProvider(
        segments=(
            {"text": transcript_text, "isFinal": True, "startMs": 0, "endMs": 500},
        ),
        events=events,
    )
    llm = FakeLlmProvider(
        chunks=({"text": "시간이 됐어요", "sequence": 0},),
        events=events,
    )
    tts = FakeTtsProvider(
        audio_by_text={"시간이 됐어요": (b"response-audio",)},
        events=events,
    )
    return ProviderBundle(llm=llm, stt=stt, tts=tts)


@pytest.mark.asyncio
async def test_runtime_connects_stt_llm_tts_in_order_and_returns_audio() -> None:
    events: list[str] = []
    bundle = create_bundle(events)
    policy = create_policy()
    runtime = ConversationRuntime(bundle)

    result = [
        chunk
        async for chunk in runtime.run("session-1", policy, audio_chunks())
    ]

    assert result == [b"response-audio"]
    assert events == ["stt", "llm", "tts"]


@pytest.mark.asyncio
async def test_runtime_does_not_synthesize_for_forbidden_topic() -> None:
    events: list[str] = []
    bundle = create_bundle(events, transcript_text="payment details")
    runtime = ConversationRuntime(bundle)

    result = [
        chunk
        async for chunk in runtime.run(
            "session-1",
            create_policy(forbidden_topics=("payment",)),
            audio_chunks(),
        )
    ]

    assert result == []
    assert events == ["stt"]


@pytest.mark.asyncio
async def test_runtime_stops_before_processing_audio_after_user_termination() -> None:
    events: list[str] = []
    bundle = create_bundle(events)
    runtime = ConversationRuntime(bundle)

    runtime.stop("session-1", "user_hangup")
    result = [
        chunk
        async for chunk in runtime.run("session-1", create_policy(), audio_chunks())
    ]

    assert result == []
    assert events == []


@pytest.mark.asyncio
async def test_runtime_emits_user_and_assistant_events_for_evaluation() -> None:
    provider_events: list[str] = []
    conversation_events: list[ConversationEvent] = []
    runtime = ConversationRuntime(
        create_bundle(provider_events),
        ConversationRuntimeOptions(event_sink=conversation_events.append),
    )

    _ = [
        chunk
        async for chunk in runtime.run("session-1", create_policy(), audio_chunks())
    ]

    assert conversation_events[0].type == "user_turn"
    assert conversation_events[0].text == "hello"
    assert conversation_events[1].type == "assistant_chunk"
    assert conversation_events[1].text == "시간이 됐어요"


@pytest.mark.asyncio
async def test_runtime_stops_before_synthesis_after_five_minutes() -> None:
    events: list[str] = []
    bundle = create_bundle(events)
    clock_values = iter((0.0, 301.0))
    runtime = ConversationRuntime(
        bundle,
        ConversationRuntimeOptions(clock=lambda: next(clock_values)),
    )

    result = [
        chunk
        async for chunk in runtime.run("session-1", create_policy(), audio_chunks())
    ]

    assert result == []
    assert events == ["stt", "llm"]
