from collections.abc import AsyncIterator

import pytest

from oncue_voice.conversation.models import AssistantChunk, DialoguePolicy, UserTurn
from oncue_voice.providers.fake_llm_provider import FakeLlmProvider
from oncue_voice.providers.fake_stt_provider import FakeSttProvider
from oncue_voice.providers.fake_tts_provider import FakeTtsProvider


async def audio_chunks() -> AsyncIterator[bytes]:
    yield b"input-audio"


async def user_turns() -> AsyncIterator[UserTurn]:
    yield UserTurn(text="hello", created_at="2026-09-10T00:00:00Z")


def create_policy() -> DialoguePolicy:
    return DialoguePolicy(
        role="friendly helper",
        stages=("greeting", "goal"),
        goal="help the child get ready for bed",
        allowed_topics=("bedtime",),
        forbidden_topics=("payment",),
        termination_conditions=("goal reached",),
        language="ko-KR",
        voice_id="fake-voice",
        instructions=("Use a warm tone.",),
        dialogue_rules=("Ask one question at a time.",),
        scenario_context="The child is getting ready for bed.",
        voice_settings={"speed": 1.0},
    )


@pytest.mark.asyncio
async def test_fake_stt_streams_configured_transcript_segments() -> None:
    provider = FakeSttProvider(
        segments=(
            {"text": "hello", "isFinal": True, "startMs": 0, "endMs": 500},
        )
    )

    result = [segment async for segment in provider.stream_transcribe(audio_chunks())]

    assert result[0].text == "hello"
    assert result[0].is_final is True


@pytest.mark.asyncio
async def test_fake_llm_streams_configured_chunks_after_reading_turns() -> None:
    provider = FakeLlmProvider(
        chunks=(AssistantChunk(text="잘 자요", sequence=0),)
    )
    policy = create_policy()

    result = [
        chunk
        async for chunk in provider.stream_reply(policy, user_turns())
    ]

    assert result == [AssistantChunk(text="잘 자요", sequence=0)]
    assert provider.last_policy == policy
    assert provider.received_turns == [
        UserTurn(text="hello", created_at="2026-09-10T00:00:00Z")
    ]


@pytest.mark.asyncio
async def test_fake_tts_streams_configured_audio_for_text() -> None:
    provider = FakeTtsProvider(audio_by_text={"안녕": (b"audio-1", b"audio-2")})

    result = [chunk async for chunk in provider.stream_synthesize("안녕")]

    assert result == [b"audio-1", b"audio-2"]
