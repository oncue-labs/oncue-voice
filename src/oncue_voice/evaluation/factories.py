from oncue_voice.evaluation.split_pipeline_evaluation_service import (
    SplitPipelineEvaluationService,
)
from oncue_voice.providers.realtime.factory import create_openai_realtime_provider
from oncue_voice.providers.realtime.fake_provider import FakeRealtimeProvider
from oncue_voice.providers.realtime.models import RealtimeEvent
from oncue_voice.providers.realtime.provider import RealtimeProvider
from oncue_voice.providers.split_pipeline.factory import (
    ProviderFactory,
    create_openai_factory,
)
from oncue_voice.providers.split_pipeline.fake_llm_provider import FakeLlmProvider
from oncue_voice.providers.split_pipeline.fake_stt_provider import FakeSttProvider
from oncue_voice.providers.split_pipeline.fake_tts_provider import FakeTtsProvider
from oncue_voice.providers.split_pipeline.models import (
    ProviderBundle,
    ProviderCapabilities,
)


def create_split_pipeline_evaluation_service(
    provider: str,
) -> SplitPipelineEvaluationService:
    if provider == "fake":
        return SplitPipelineEvaluationService(create_fake_factory())
    if provider == "openai":
        return SplitPipelineEvaluationService(create_openai_factory())
    raise ValueError(f"Unsupported evaluation provider: {provider}")


def create_realtime_provider(provider: str) -> RealtimeProvider:
    if provider == "fake":
        return FakeRealtimeProvider(
            (
                RealtimeEvent(type="speech_started"),
                RealtimeEvent(type="audio_delta", audio=b"\x00\x00" * 240),
                RealtimeEvent(
                    type="transcript_completed",
                    text="합성 테스트 발화입니다.",
                ),
                RealtimeEvent(type="response_completed"),
            )
        )
    if provider == "openai":
        return create_openai_realtime_provider()
    raise ValueError(f"Unsupported realtime provider: {provider}")


def create_fake_factory() -> ProviderFactory:
    factory = ProviderFactory()
    factory.register(
        "fake",
        ProviderCapabilities(
            provider="fake",
            llm_models=("fake-llm",),
            stt_models=("fake-stt",),
            tts_voice_ids=("fake-voice",),
            tts_options=("speed", "instructions"),
        ),
        lambda settings: create_fake_bundle(),
    )
    return factory


def create_fake_bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=FakeLlmProvider(
            chunks=({"text": "평가용 응답입니다.", "sequence": 0},)
        ),
        stt=FakeSttProvider(
            segments=(
                {
                    "text": "합성 테스트 발화입니다.",
                    "isFinal": True,
                    "startMs": 0,
                    "endMs": 500,
                },
            )
        ),
        tts=FakeTtsProvider(
            audio_by_text={"평가용 응답입니다.": (b"\x00\x00" * 240,)}
        ),
    )
