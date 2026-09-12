import base64
import binascii
from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.conversation.policy_renderer import render_dialogue_policy
from oncue_voice.providers.common.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from oncue_voice.providers.realtime.models import (
    RealtimeEvent,
    RealtimeSessionOptions,
)


class OpenAiRealtimeProvider:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def connect(
        self,
        policy: DialoguePolicy,
        options: RealtimeSessionOptions,
    ) -> "OpenAiRealtimeSession":
        session_config = self._session_config(policy, options)
        try:
            connection_manager = self._realtime_api().connect(model=options.model)
            connection = await self._enter_connection(connection_manager)
            await connection.send(
                {
                    "type": "session.update",
                    "session": session_config,
                }
            )
            return OpenAiRealtimeSession(connection)
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

    def _realtime_api(self) -> Any:
        realtime_api = getattr(self._client, "realtime", None)
        if realtime_api is not None:
            return realtime_api
        beta_api = getattr(self._client, "beta", None)
        if beta_api is not None:
            return beta_api.realtime
        raise ProviderRequestError("OpenAI Realtime API is unavailable")

    @staticmethod
    async def _enter_connection(connection_manager: Any) -> Any:
        enter = getattr(connection_manager, "enter", None)
        if enter is not None:
            return await enter()
        return await connection_manager.__aenter__()

    @staticmethod
    def _session_config(
        policy: DialoguePolicy,
        options: RealtimeSessionOptions,
    ) -> dict[str, Any]:
        return {
            "type": "realtime",
            "model": options.model,
            "instructions": render_dialogue_policy(policy),
            "audio": {
                "input": {
                    "format": OpenAiRealtimeProvider._audio_format(
                        options.input_audio_format,
                        options.sample_rate_hz,
                    ),
                    "turn_detection": {"type": options.turn_detection},
                },
                "output": {
                    "format": OpenAiRealtimeProvider._audio_format(
                        options.output_audio_format,
                        options.sample_rate_hz,
                    ),
                    "voice": options.voice_id,
                },
            },
        }

    @staticmethod
    def _audio_format(format_name: str, sample_rate_hz: int) -> dict[str, Any]:
        if format_name == "pcm16":
            return {"type": "audio/pcm", "rate": sample_rate_hz}
        if format_name == "pcmu":
            return {"type": "audio/pcmu"}
        if format_name == "pcma":
            return {"type": "audio/pcma"}
        raise ProviderResponseError(
            f"Unsupported OpenAI Realtime audio format: {format_name}"
        )


class OpenAiRealtimeSession:
    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._closed = False

    async def send_audio_chunk(self, audio: bytes) -> None:
        if not audio:
            return
        encoded_audio = base64.b64encode(audio).decode("ascii")
        try:
            await self._connection.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": encoded_audio,
                }
            )
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

    def events(self) -> AsyncIterator[RealtimeEvent]:
        return self._event_stream()

    async def _event_stream(self) -> AsyncIterator[RealtimeEvent]:
        try:
            async for event in self._connection:
                mapped_event = self._to_event(event)
                if mapped_event is not None:
                    yield mapped_event
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

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._connection.close()

    @classmethod
    def _to_event(cls, event: Any) -> RealtimeEvent | None:
        event_type = cls._value(event, "type")
        if event_type == "response.output_audio.delta":
            delta = cls._required_string(event, "delta")
            try:
                audio = base64.b64decode(delta, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ProviderResponseError(
                    "OpenAI Realtime audio event is invalid"
                ) from error
            return RealtimeEvent(type="audio_delta", audio=audio)
        if event_type in {
            "response.output_audio_transcript.delta",
            "response.output_text.delta",
        }:
            return RealtimeEvent(
                type="transcript_delta",
                text=cls._required_string(event, "delta"),
            )
        if event_type == "conversation.item.input_audio_transcription.completed":
            return RealtimeEvent(
                type="transcript_completed",
                text=cls._required_string(event, "transcript"),
            )
        if event_type == "input_audio_buffer.speech_started":
            return RealtimeEvent(type="speech_started")
        if event_type == "input_audio_buffer.speech_stopped":
            return RealtimeEvent(type="speech_stopped")
        if event_type == "response.done":
            return RealtimeEvent(type="response_completed")
        if event_type == "error":
            error_details = cls._value(event, "error")
            message = cls._value(error_details, "message") or cls._value(
                event,
                "message",
            )
            return RealtimeEvent(
                type="error",
                message=message if isinstance(message, str) else "OpenAI Realtime error",
            )
        return None

    @staticmethod
    def _value(value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)

    @classmethod
    def _required_string(cls, value: Any, name: str) -> str:
        result = cls._value(value, name)
        if not isinstance(result, str):
            raise ProviderResponseError(
                f"OpenAI Realtime event field is invalid: {name}"
            )
        return result
