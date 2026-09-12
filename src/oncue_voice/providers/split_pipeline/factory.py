import os
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI

from oncue_voice.providers.split_pipeline.models import (
    ProviderBundle,
    ProviderCapabilities,
    ProviderRegistration,
    ProviderSettings,
)
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


class UnsupportedProviderError(ValueError):
    """Raised when no adapter is registered for a provider name."""


class UnsupportedProviderSettingError(ValueError):
    """Raised when settings are not supported by a provider capability."""


ProviderBuilder = Callable[[ProviderSettings], ProviderBundle]


class ProviderFactory:
    def __init__(self) -> None:
        self._registrations: dict[str, ProviderRegistration] = {}

    def register(
        self,
        provider: str,
        capabilities: ProviderCapabilities,
        builder: ProviderBuilder,
    ) -> None:
        self._registrations[provider] = ProviderRegistration(
            capabilities=capabilities,
            builder=builder,
        )

    def create(self, settings: ProviderSettings) -> ProviderBundle:
        registration = self._registrations.get(settings.provider)
        if registration is None:
            raise UnsupportedProviderError(
                f"Unsupported provider: {settings.provider}"
            )

        if registration.capabilities.provider != settings.provider:
            raise UnsupportedProviderError(
                f"Provider capability mismatch: {settings.provider}"
            )

        unsupported = registration.capabilities.unsupported_settings(settings)
        if unsupported:
            fields = ", ".join(unsupported)
            raise UnsupportedProviderSettingError(
                f"Unsupported provider settings: {fields}"
            )

        return registration.builder(settings)


def create_openai_factory(
    client: Any | None = None,
    api_key: str | None = None,
) -> ProviderFactory:
    resolved_client = client
    if resolved_client is None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        resolved_client = AsyncOpenAI(api_key=resolved_api_key)

    factory = ProviderFactory()
    factory.register(
        "openai",
        ProviderCapabilities(
            provider="openai",
            llm_models=("gpt-5-mini", "gpt-5.5"),
            stt_models=("gpt-4o-mini-transcribe", "gpt-4o-transcribe"),
            tts_voice_ids=(
                "alloy",
                "ash",
                "coral",
                "echo",
                "fable",
                "nova",
                "onyx",
                "sage",
                "shimmer",
                "verse",
            ),
            tts_options=("speed", "instructions"),
        ),
        lambda settings: ProviderBundle(
            llm=OpenAiLlmProvider(
                resolved_client,
                OpenAiLlmSettings(model=settings.llm_model),
            ),
            stt=OpenAiSttProvider(
                resolved_client,
                OpenAiSttSettings(model=settings.stt_model),
            ),
            tts=OpenAiTtsProvider(
                resolved_client,
                OpenAiTtsSettings(
                    model="gpt-4o-mini-tts",
                    voice_id=settings.tts_voice_id,
                    speed=settings.tts_options.get("speed"),
                    instructions=settings.tts_options.get("instructions"),
                ),
            ),
        ),
    )
    return factory
