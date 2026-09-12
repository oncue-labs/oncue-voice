from collections.abc import AsyncIterator
from typing import Protocol

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.providers.realtime.models import (
    RealtimeEvent,
    RealtimeSessionOptions,
)


class RealtimeSession(Protocol):
    async def send_audio_chunk(self, audio: bytes) -> None:
        """Send one ordered audio chunk to the speech-to-speech provider."""

    def events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield provider-neutral output and state events."""

    async def close(self) -> None:
        """Close the provider connection once."""


class RealtimeProvider(Protocol):
    async def connect(
        self,
        policy: DialoguePolicy,
        options: RealtimeSessionOptions,
    ) -> RealtimeSession:
        """Open a session configured with the final dialogue policy."""
