import pytest

from oncue_voice.providers.split_pipeline.factory import (
    ProviderFactory,
    UnsupportedProviderError,
    UnsupportedProviderSettingError,
)
from oncue_voice.providers.split_pipeline.models import (
    ProviderBundle,
    ProviderCapabilities,
    ProviderSettings,
)


class StubLlmProvider:
    async def stream_reply(self, policy, turns):
        raise NotImplementedError


class StubSttProvider:
    async def stream_transcribe(self, audio):
        raise NotImplementedError


class StubTtsProvider:
    async def stream_synthesize(self, text):
        raise NotImplementedError


def create_registered_factory() -> ProviderFactory:
    capabilities = ProviderCapabilities(
        provider="stub",
        llm_models=("stub-llm",),
        stt_models=("stub-stt",),
        tts_voice_ids=("stub-voice",),
        tts_options=("speed",),
    )
    bundle = ProviderBundle(
        llm=StubLlmProvider(),
        stt=StubSttProvider(),
        tts=StubTtsProvider(),
    )

    factory = ProviderFactory()
    factory.register("stub", capabilities, lambda settings: bundle)
    return factory


def test_factory_returns_bundle_for_supported_settings() -> None:
    factory = create_registered_factory()
    settings = ProviderSettings(
        provider="stub",
        llm_model="stub-llm",
        stt_model="stub-stt",
        tts_voice_id="stub-voice",
        tts_options={"speed": 1.0},
    )

    result = factory.create(settings)

    assert isinstance(result, ProviderBundle)
    assert result.llm.__class__ is StubLlmProvider


def test_factory_rejects_unknown_provider() -> None:
    factory = create_registered_factory()
    settings = ProviderSettings(
        provider="unknown",
        llm_model="stub-llm",
        stt_model="stub-stt",
        tts_voice_id="stub-voice",
    )

    with pytest.raises(UnsupportedProviderError):
        factory.create(settings)


def test_factory_rejects_setting_outside_provider_capabilities() -> None:
    factory = create_registered_factory()
    settings = ProviderSettings(
        provider="stub",
        llm_model="stub-llm",
        stt_model="stub-stt",
        tts_voice_id="unlisted-voice",
        tts_options={"emotion": "happy"},
    )

    with pytest.raises(UnsupportedProviderSettingError):
        factory.create(settings)
