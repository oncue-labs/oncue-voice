import json

import pytest

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.evaluation.realtime_evaluation_service import (
    RealtimeEvaluationRequest,
    RealtimeEvaluationService,
)
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
        allowed_topics=("bedtime",),
        forbidden_topics=("payment",),
        termination_conditions=("child is ready for bed",),
        language="ko-KR",
        voice_id="fake-voice",
        instructions=("Speak warmly.",),
        dialogue_rules=("Ask one question at a time.",),
        scenario_context="The child is preparing for bed.",
    )


@pytest.mark.asyncio
async def test_realtime_evaluation_service_writes_audio_and_transcript_artifacts(
    tmp_path,
) -> None:
    provider = FakeRealtimeProvider(
        (
            RealtimeEvent(type="transcript_completed", text="잘 준비했어요"),
            RealtimeEvent(type="audio_delta", audio=b"pcm-audio"),
            RealtimeEvent(type="response_completed"),
        )
    )
    service = RealtimeEvaluationService(provider)
    request = RealtimeEvaluationRequest(
        run_id="run-001",
        variant_id="variant-a",
        combination_key="santa-child-roleplay",
        input_text="잘 준비했어요",
        policy=create_policy(),
        session_options=RealtimeSessionOptions(
            model="fake-realtime",
            voice_id="fake-voice",
        ),
        input_audio=b"synthetic-input-audio",
        artifact_root=tmp_path,
        provider="fake",
    )

    result = await service.run(request)

    assert result.succeeded is True
    assert (result.artifact_directory / "response.wav").read_bytes().startswith(b"RIFF")
    run_data = json.loads((result.artifact_directory / "run.json").read_text())
    assert run_data["evaluationPath"] == "realtime"
    assert run_data["audioBytes"] == len(b"pcm-audio")
    transcript_data = json.loads(
        (result.artifact_directory / "transcript.json").read_text()
    )
    assert transcript_data["events"] == [
        {
            "type": "transcript_completed",
            "text": "잘 준비했어요",
            "message": None,
        },
        {
            "type": "audio_delta",
            "text": None,
            "message": None,
        },
        {
            "type": "response_completed",
            "text": None,
            "message": None,
        },
    ]


@pytest.mark.asyncio
async def test_realtime_evaluation_service_marks_provider_error_as_failed(tmp_path) -> None:
    service = RealtimeEvaluationService(
        FakeRealtimeProvider((RealtimeEvent(type="error", message="failed"),))
    )
    request = RealtimeEvaluationRequest(
        run_id="run-002",
        variant_id="variant-a",
        combination_key="santa-child-roleplay",
        input_text="synthetic input",
        policy=create_policy(),
        session_options=RealtimeSessionOptions(
            model="fake-realtime",
            voice_id="fake-voice",
        ),
        input_audio=b"synthetic-input-audio",
        artifact_root=tmp_path,
        provider="fake",
    )

    result = await service.run(request)

    assert result.succeeded is False
