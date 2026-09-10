from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from oncue_voice.providers.llm_provider import LlmProvider
from oncue_voice.providers.stt_provider import SttProvider
from oncue_voice.providers.tts_provider import TtsProvider


class ProviderSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider: str
    llm_model: str = Field(alias="llmModel")
    stt_model: str = Field(alias="sttModel")
    tts_voice_id: str = Field(alias="ttsVoiceId")
    tts_options: dict[str, Any] = Field(
        default_factory=dict,
        alias="ttsOptions",
    )


class ProviderCapabilities(BaseModel):
    provider: str
    llm_models: tuple[str, ...] = ()
    stt_models: tuple[str, ...] = ()
    tts_voice_ids: tuple[str, ...] = ()
    tts_options: tuple[str, ...] = ()

    def unsupported_settings(self, settings: ProviderSettings) -> list[str]:
        unsupported: list[str] = []
        if settings.llm_model not in self.llm_models:
            unsupported.append("llm_model")
        if settings.stt_model not in self.stt_models:
            unsupported.append("stt_model")
        if settings.tts_voice_id not in self.tts_voice_ids:
            unsupported.append("tts_voice_id")
        unsupported.extend(
            f"tts_options.{option}"
            for option in settings.tts_options
            if option not in self.tts_options
        )
        return unsupported


@dataclass(frozen=True)
class ProviderBundle:
    llm: LlmProvider
    stt: SttProvider
    tts: TtsProvider


@dataclass(frozen=True)
class ProviderRegistration:
    capabilities: ProviderCapabilities
    builder: Callable[[ProviderSettings], ProviderBundle]
