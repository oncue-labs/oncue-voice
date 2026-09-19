import base64
from types import SimpleNamespace

import pytest

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.providers.common.errors import ProviderResponseError
from oncue_voice.providers.realtime.models import RealtimeSessionOptions
from oncue_voice.providers.realtime.openai_provider import OpenAiRealtimeProvider


class FakeRealtimeConnection:
    def __init__(self, events):
        self.events = events
        self.sent = []
        self.closed = False

    async def send(self, event):
        self.sent.append(event)

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for event in self.events:
            yield event

    async def close(self):
        self.closed = True


class FakeRealtimeConnectionManager:
    def __init__(self, connection):
        self.connection = connection

    async def enter(self):
        return self.connection


class FakeOpenAiClient:
    def __init__(self, connection):
        self.connection = connection
        self.realtime = SimpleNamespace(connect=self.connect)

    def connect(self, **kwargs):
        self.model = kwargs["model"]
        return FakeRealtimeConnectionManager(self.connection)


def create_policy() -> DialoguePolicy:
    return DialoguePolicy(
        role="friendly helper",
        stages=("greeting",),
        goal="help the child get ready for bed",
        allowed_topics=("bedtime",),
        forbidden_topics=("payment",),
        language="ko-KR",
        voice_id="alloy",
        instructions=("Use a warm tone.",),
        dialogue_rules=("Ask one question at a time.",),
        scenario_context="The child is getting ready for bed.",
    )


@pytest.mark.asyncio
async def test_openai_realtime_adapter_sends_policy_and_audio_event() -> None:
    connection = FakeRealtimeConnection(
        (
            SimpleNamespace(type="session.created"),
            SimpleNamespace(
                type="response.output_audio.delta",
                delta=base64.b64encode(b"audio").decode(),
            ),
            SimpleNamespace(
                type="response.output_audio_transcript.delta",
                delta="안녕하세요",
            ),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="잘 시간이야",
            ),
            SimpleNamespace(type="response.done"),
        )
    )
    client = FakeOpenAiClient(connection)
    provider = OpenAiRealtimeProvider(client)
    options = RealtimeSessionOptions(
        model="gpt-realtime",
        voice_id="santa-default",
        sample_rate_hz=24_000,
    )

    session = await provider.connect(create_policy(), options)
    await session.send_audio_chunk(b"input-audio")
    events = [event async for event in session.events()]

    assert client.model == "gpt-realtime"
    assert connection.sent[0]["type"] == "session.update"
    session_config = connection.sent[0]["session"]
    assert session_config["instructions"]
    assert session_config["output_modalities"] == ["audio"]
    assert session_config["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24_000,
    }
    assert session_config["audio"]["input"]["turn_detection"] == {
        "type": "server_vad",
        "create_response": True,
        "interrupt_response": True,
    }
    assert session_config["audio"]["output"]["format"] == {
        "type": "audio/pcm",
        "rate": 24_000,
    }
    assert session_config["audio"]["output"]["voice"] == "alloy"
    assert connection.sent[1] == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(b"input-audio").decode(),
    }
    assert events[0].type == "audio_delta"
    assert events[0].audio == b"audio"
    assert events[1].type == "transcript_delta"
    assert events[1].text == "안녕하세요"
    assert events[2].type == "transcript_completed"
    assert events[2].text == "잘 시간이야"
    assert events[3].type == "response_completed"


@pytest.mark.asyncio
async def test_openai_realtime_adapter_maps_provider_error_event() -> None:
    connection = FakeRealtimeConnection(
        (
            SimpleNamespace(
                type="error",
                error=SimpleNamespace(message="request rejected"),
            ),
        )
    )
    provider = OpenAiRealtimeProvider(FakeOpenAiClient(connection))
    session = await provider.connect(
        create_policy(),
        RealtimeSessionOptions(model="gpt-realtime", voice_id="alloy"),
    )

    events = [event async for event in session.events()]

    assert events[0].type == "error"
    assert events[0].message == "request rejected"


@pytest.mark.asyncio
async def test_openai_realtime_adapter_rejects_unknown_audio_format() -> None:
    connection = FakeRealtimeConnection(())
    provider = OpenAiRealtimeProvider(FakeOpenAiClient(connection))

    with pytest.raises(ProviderResponseError, match="audio format"):
        await provider.connect(
            create_policy(),
            RealtimeSessionOptions(
                model="gpt-realtime",
                voice_id="alloy",
                input_audio_format="unknown",
            ),
        )
