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

from oncue_voice.providers.common.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
)


@dataclass(frozen=True)
class OpenAiTtsSettings:
    model: str
    voice_id: str
    response_format: str = "pcm"
    speed: float | None = None
    instructions: str | None = None
    timeout: float | None = None


class OpenAiTtsProvider:
    def __init__(self, client: Any, settings: OpenAiTtsSettings) -> None:
        self._client = client
        self._settings = settings

    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        request: dict[str, Any] = {
            "input": text,
            "model": self._settings.model,
            "voice": self._settings.voice_id,
            "response_format": self._settings.response_format,
        }
        if self._settings.speed is not None:
            request["speed"] = self._settings.speed
        if self._settings.instructions is not None:
            request["instructions"] = self._settings.instructions
        if self._settings.timeout is not None:
            request["timeout"] = self._settings.timeout

        try:
            async with self._client.audio.speech.with_streaming_response.create(
                **request
            ) as response:
                async for chunk in response.iter_bytes():
                    yield chunk
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
