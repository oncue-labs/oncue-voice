from oncue_voice.evaluation.service import EvaluationService
from oncue_voice.providers.fake_llm_provider import FakeLlmProvider
from oncue_voice.providers.fake_stt_provider import FakeSttProvider
from oncue_voice.providers.fake_tts_provider import FakeTtsProvider
from oncue_voice.providers.factory import (
    ProviderFactory,
    create_openai_factory,
)
from oncue_voice.providers.models import ProviderBundle, ProviderCapabilities


def create_evaluation_service(provider: str) -> EvaluationService:
    if provider == "fake":
        return EvaluationService(create_fake_factory())
    if provider == "openai":
        return EvaluationService(create_openai_factory())
    raise ValueError(f"Unsupported evaluation provider: {provider}")


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
