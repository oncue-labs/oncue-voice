import base64
import binascii
import logging
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


logger = logging.getLogger(__name__)


class OpenAiRealtimeProvider:
    _VOICE_ALIASES = {
        "santa-default": "alloy",
        "princess-default": "coral",
        "friend-default": "ash",
    }

    def __init__(self, client: Any) -> None:
        self._client = client

    async def connect(
        self,
        policy: DialoguePolicy,
        options: RealtimeSessionOptions,
    ) -> "OpenAiRealtimeSession":
        session_config = self._session_config(policy, options)
        try:
            logger.info(
                "openai.realtime_connect_started model=%s",
                options.model,
            )
            connection_manager = self._realtime_api().connect(model=options.model)
            connection = await self._enter_connection(connection_manager)
            await connection.send(
                {
                    "type": "session.update",
                    "session": session_config,
                }
            )
            logger.info(
                "openai.realtime_session_configured model=%s",
                options.model,
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
            "output_modalities": ["audio"],
            "instructions": render_dialogue_policy(policy),
            "audio": {
                "input": {
                    "format": OpenAiRealtimeProvider._audio_format(
                        options.input_audio_format,
                        options.sample_rate_hz,
                    ),
                    "turn_detection": {
                        "type": options.turn_detection,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": OpenAiRealtimeProvider._audio_format(
                        options.output_audio_format,
                        options.sample_rate_hz,
                    ),
                    "voice": OpenAiRealtimeProvider._provider_voice_id(
                        options.voice_id
                    ),
                },
            },
        }

    @classmethod
    def _provider_voice_id(cls, voice_id: str) -> str:
        """Translate OnCue's stable persona voice ID to OpenAI's voice ID."""
        return cls._VOICE_ALIASES.get(voice_id, voice_id)

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
        self._input_chunk_count = 0
        self._output_audio_chunk_count = 0

    async def send_audio_chunk(self, audio: bytes) -> None:
        if not audio:
            return
        self._input_chunk_count += 1
        if self._input_chunk_count == 1:
            logger.info(
                "openai.realtime_input_audio_started bytes=%s",
                len(audio),
            )
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
                event_type = self._value(event, "type")
                logger.info("openai.realtime_event type=%s", event_type)
                mapped_event = self._to_event(event)
                if mapped_event is not None:
                    if mapped_event.type == "audio_delta":
                        self._output_audio_chunk_count += 1
                        if self._output_audio_chunk_count == 1:
                            logger.info("openai.realtime_output_audio_started")
                    elif mapped_event.type == "error":
                        logger.warning(
                            "openai.realtime_error_received message=%s",
                            mapped_event.message,
                        )
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
