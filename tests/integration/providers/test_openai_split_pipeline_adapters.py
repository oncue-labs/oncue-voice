from types import SimpleNamespace

import pytest

from oncue_voice.conversation.models import AssistantChunk, DialoguePolicy, UserTurn
from oncue_voice.providers.split_pipeline.openai_llm_provider import (
    OpenAiLlmProvider,
    OpenAiLlmSettings,
)
from oncue_voice.providers.split_pipeline.openai_stt_provider import (
    OpenAiSttProvider,
    OpenAiSttSettings,
)
from oncue_voice.providers.split_pipeline.openai_tts_provider import (
    OpenAiTtsProvider,
    OpenAiTtsSettings,
)
from oncue_voice.providers.common.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from oncue_voice.providers.split_pipeline.factory import create_openai_factory
from oncue_voice.providers.split_pipeline.models import ProviderSettings


async def events_stream(events):
    for event in events:
        yield event


async def audio_stream():
    yield b"audio-part-1"
    yield b"audio-part-2"


def create_policy() -> DialoguePolicy:
    return DialoguePolicy(
        role="friendly helper",
        stages=("greeting",),
        goal="help the child get ready for bed",
        allowed_topics=("bedtime",),
        forbidden_topics=("payment",),
        termination_conditions=("goal reached",),
        language="ko-KR",
        voice_id="alloy",
        instructions=("Use a warm tone.",),
        dialogue_rules=("Ask one question at a time.",),
        scenario_context="The child is getting ready for bed.",
        voice_settings={"speed": 1.0},
    )


class FakeChatCompletions:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error
        self.request = None

    async def create(self, **request):
        self.request = request
        if self.error is not None:
            raise self.error
        return events_stream(self.events)


class FakeTranscriptions:
    def __init__(self, events):
        self.events = events
        self.request = None

    async def create(self, **request):
        self.request = request
        return events_stream(self.events)


class FakeSpeechResponse:
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def iter_bytes(self):
        for chunk in self.chunks:
            yield chunk


class FakeSpeech:
    def __init__(self, chunks):
        self.chunks = chunks
        self.request = None
        self.with_streaming_response = self

    def create(self, **request):
        self.request = request
        return FakeSpeechResponse(self.chunks)


class FakeOpenAiClient:
    def __init__(self, chat_events=(), transcription_events=(), speech_chunks=()):
        self.chat = SimpleNamespace(
            completions=FakeChatCompletions(list(chat_events))
        )
        self.audio = SimpleNamespace(
            transcriptions=FakeTranscriptions(list(transcription_events)),
            speech=FakeSpeech(list(speech_chunks)),
        )


def test_openai_factory_builds_three_adapters_from_provider_settings() -> None:
    client = FakeOpenAiClient()
    factory = create_openai_factory(client=client)

    bundle = factory.create(
        ProviderSettings(
            provider="openai",
            llm_model="gpt-5-mini",
            stt_model="gpt-4o-mini-transcribe",
            tts_voice_id="alloy",
            tts_options={"speed": 1.0},
        )
    )

    assert bundle.llm.__class__.__name__ == "OpenAiLlmProvider"
    assert bundle.stt.__class__.__name__ == "OpenAiSttProvider"
    assert bundle.tts.__class__.__name__ == "OpenAiTtsProvider"


@pytest.mark.asyncio
async def test_llm_adapter_sends_policy_and_turns_and_streams_text_chunks() -> None:
    client = FakeOpenAiClient(
        chat_events=(
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="안녕"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="하세요"))]
            ),
        )
    )
    provider = OpenAiLlmProvider(
        client,
        OpenAiLlmSettings(model="gpt-5-mini", temperature=0.2),
    )

    result = [
        chunk
        async for chunk in provider.stream_reply(
            create_policy(),
            events_stream((UserTurn(text="잘 시간이야"),)),
        )
    ]

    assert result == [
        AssistantChunk(text="안녕", sequence=0),
        AssistantChunk(text="하세요", sequence=1),
    ]
    request = client.chat.completions.request
    assert request["model"] == "gpt-5-mini"
    assert request["stream"] is True
    assert request["temperature"] == 0.2
    assert request["messages"][-1] == {
        "role": "user",
        "content": "잘 시간이야",
    }
    assert "friendly helper" in request["messages"][0]["content"]


@pytest.mark.asyncio
async def test_stt_adapter_uploads_audio_and_streams_final_transcript() -> None:
    client = FakeOpenAiClient(
        transcription_events=(
            SimpleNamespace(type="transcript.text.delta", delta="안녕"),
            SimpleNamespace(
                type="transcript.text.done",
                text="안녕",
            ),
        )
    )
    provider = OpenAiSttProvider(
        client,
        OpenAiSttSettings(model="gpt-4o-mini-transcribe", language="ko"),
    )

    result = [segment async for segment in provider.stream_transcribe(audio_stream())]

    assert result[-1].text == "안녕"
    assert result[-1].is_final is True
    request = client.audio.transcriptions.request
    assert request["model"] == "gpt-4o-mini-transcribe"
    assert request["language"] == "ko"
    assert request["stream"] is True
    assert request["file"][1] == b"audio-part-1audio-part-2"


@pytest.mark.asyncio
async def test_tts_adapter_streams_pcm_bytes_with_voice_settings() -> None:
    client = FakeOpenAiClient(speech_chunks=(b"pcm-1", b"pcm-2"))
    provider = OpenAiTtsProvider(
        client,
        OpenAiTtsSettings(
            model="gpt-4o-mini-tts",
            voice_id="alloy",
            speed=0.9,
            instructions="Speak warmly.",
        ),
    )

    result = [chunk async for chunk in provider.stream_synthesize("안녕")]

    assert result == [b"pcm-1", b"pcm-2"]
    assert client.audio.speech.request == {
        "input": "안녕",
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "response_format": "pcm",
        "speed": 0.9,
        "instructions": "Speak warmly.",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("authentication", ProviderAuthenticationError),
        ("rate_limit", ProviderRateLimitError),
        ("timeout", ProviderTimeoutError),
    ],
)
async def test_llm_adapter_maps_openai_failures_to_provider_errors(
    error,
    expected,
) -> None:
    from openai import APITimeoutError, AuthenticationError, RateLimitError

    import httpx

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    authentication_response = httpx.Response(401, request=request)
    rate_limit_response = httpx.Response(429, request=request)
    errors = {
        "authentication": AuthenticationError(
            "bad key",
            response=authentication_response,
            body={},
        ),
        "rate_limit": RateLimitError(
            "slow down",
            response=rate_limit_response,
            body={},
        ),
        "timeout": APITimeoutError(request=request),
    }
    client = FakeOpenAiClient()
    client.chat.completions.error = errors[error]
    provider = OpenAiLlmProvider(
        client,
        OpenAiLlmSettings(model="gpt-5-mini"),
    )

    with pytest.raises(expected):
        _ = [
            chunk
            async for chunk in provider.stream_reply(
                create_policy(),
                events_stream((UserTurn(text="hello"),)),
            )
        ]
