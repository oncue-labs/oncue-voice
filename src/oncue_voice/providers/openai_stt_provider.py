from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from oncue_voice.conversation.models import TranscriptSegment
from oncue_voice.providers.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
)


@dataclass(frozen=True)
class OpenAiSttSettings:
    model: str
    language: str | None = None
    timeout: float | None = None
    file_name: str = "oncue-audio.wav"
    content_type: str = "audio/wav"


class OpenAiSttProvider:
    def __init__(self, client: Any, settings: OpenAiSttSettings) -> None:
        self._client = client
        self._settings = settings

    async def stream_transcribe(
        self,
        audio: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptSegment]:
        audio_bytes = b"".join([chunk async for chunk in audio])
        if not audio_bytes:
            return

        request: dict[str, Any] = {
            "file": (
                self._settings.file_name,
                audio_bytes,
                self._settings.content_type,
            ),
            "model": self._settings.model,
            "response_format": "json",
            "stream": True,
        }
        if self._settings.language is not None:
            request["language"] = self._settings.language
        if self._settings.timeout is not None:
            request["timeout"] = self._settings.timeout

        try:
            response = await self._client.audio.transcriptions.create(**request)
            if hasattr(response, "__aiter__"):
                async for event in response:
                    yield self._to_segment(event)
                return
            yield self._to_final_segment(response)
        except AuthenticationError as error:
            raise ProviderAuthenticationError("OpenAI authentication failed") from error
        except RateLimitError as error:
            raise ProviderRateLimitError("OpenAI rate limit exceeded") from error
        except APITimeoutError as error:
            raise ProviderTimeoutError("OpenAI request timed out") from error
        except APIConnectionError as error:
            raise ProviderRequestError("OpenAI connection failed") from error
        except APIError as error:
            raise ProviderRequestError("OpenAI request failed") from error

    @staticmethod
    def _to_segment(event: Any) -> TranscriptSegment:
        event_type = getattr(event, "type", "")
        if event_type.endswith(".delta"):
            text = getattr(event, "delta", None)
            if not isinstance(text, str):
                raise ProviderResponseError("OpenAI transcript delta is invalid")
            return TranscriptSegment(
                text=text,
                is_final=False,
                start_ms=0,
                end_ms=0,
            )
        if event_type.endswith(".done"):
            text = getattr(event, "text", None)
            if not isinstance(text, str):
                raise ProviderResponseError("OpenAI transcript result is invalid")
            return TranscriptSegment(
                text=text,
                is_final=True,
                start_ms=0,
                end_ms=0,
            )
        raise ProviderResponseError("OpenAI transcript event is unsupported")

    @staticmethod
    def _to_final_segment(response: Any) -> TranscriptSegment:
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            raise ProviderResponseError("OpenAI transcription response is invalid")
        return TranscriptSegment(
            text=text,
            is_final=True,
            start_ms=0,
            end_ms=0,
        )
