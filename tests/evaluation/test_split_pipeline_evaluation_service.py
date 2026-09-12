import json

import pytest

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.evaluation.split_pipeline_evaluation_service import (
    SplitPipelineEvaluationRequest,
    SplitPipelineEvaluationService,
)
from oncue_voice.providers.split_pipeline.fake_llm_provider import FakeLlmProvider
from oncue_voice.providers.split_pipeline.fake_stt_provider import FakeSttProvider
from oncue_voice.providers.split_pipeline.fake_tts_provider import FakeTtsProvider
from oncue_voice.providers.split_pipeline.factory import ProviderFactory
from oncue_voice.providers.split_pipeline.models import (
    ProviderBundle,
    ProviderCapabilities,
    ProviderSettings,
)


def create_policy() -> DialoguePolicy:
    return DialoguePolicy(
        role="Santa",
        stages=("greeting", "goal"),
        goal="help the child get ready for bed",
        allowed_topics=("bedtime", "good behavior"),
        forbidden_topics=("payment",),
        termination_conditions=("child is ready for bed",),
        language="ko-KR",
        voice_id="fake-voice",
        instructions=("Speak warmly.",),
        dialogue_rules=("Ask one question at a time.",),
        scenario_context="The child is preparing for bed.",
        voice_settings={"speed": 1.0},
    )


def create_fake_factory() -> ProviderFactory:
    bundle = ProviderBundle(
        llm=FakeLlmProvider(chunks=({"text": "잘 자요", "sequence": 0},)),
        stt=FakeSttProvider(
            segments=(
                {"text": "잘 준비했어요", "isFinal": True, "startMs": 0, "endMs": 500},
            )
        ),
        tts=FakeTtsProvider(audio_by_text={"잘 자요": (b"pcm-audio",)}),
    )
    factory = ProviderFactory()
    factory.register(
        "fake",
        ProviderCapabilities(
            provider="fake",
            llm_models=("fake-llm",),
            stt_models=("fake-stt",),
            tts_voice_ids=("fake-voice",),
        ),
        lambda settings: bundle,
    )
    return factory


@pytest.mark.asyncio
async def test_split_pipeline_evaluation_service_writes_reproducible_local_artifacts(tmp_path) -> None:
    service = SplitPipelineEvaluationService(create_fake_factory())
    request = SplitPipelineEvaluationRequest(
        run_id="run-001",
        variant_id="variant-a",
        combination_key="santa-child-roleplay",
        input_text="잘 준비했어요",
        policy=create_policy(),
        provider_settings=ProviderSettings(
            provider="fake",
            llm_model="fake-llm",
            stt_model="fake-stt",
            tts_voice_id="fake-voice",
        ),
        input_audio=b"synthetic-input-audio",
        artifact_root=tmp_path,
    )

    result = await service.run(request)

    assert result.artifact_directory == tmp_path / "run-001" / "variant-a"
    expected_files = {
        "run.json",
        "policy-snapshot.json",
        "provider-config.json",
        "input.json",
        "transcript.json",
        "response.wav",
        "evaluation.md",
    }
    assert {path.name for path in result.artifact_directory.iterdir()} == expected_files

    run_data = json.loads((result.artifact_directory / "run.json").read_text())
    assert run_data["runId"] == "run-001"
    assert run_data["variantId"] == "variant-a"
    assert run_data["succeeded"] is True

    policy_data = json.loads(
        (result.artifact_directory / "policy-snapshot.json").read_text()
    )
    assert policy_data["goal"] == "help the child get ready for bed"
    assert policy_data["scenarioContext"] == "The child is preparing for bed."

    provider_data = json.loads(
        (result.artifact_directory / "provider-config.json").read_text()
    )
    assert provider_data["provider"] == "fake"
    assert "apiKey" not in json.dumps(provider_data)

    transcript_data = json.loads(
        (result.artifact_directory / "transcript.json").read_text()
    )
    assert transcript_data["events"] == [
        {"type": "user_turn", "text": "잘 준비했어요", "sequence": None},
        {"type": "assistant_chunk", "text": "잘 자요", "sequence": 0},
    ]
    assert (result.artifact_directory / "response.wav").read_bytes().startswith(b"RIFF")
