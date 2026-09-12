from collections.abc import AsyncIterator, Iterable

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.providers.realtime.models import (
    RealtimeEvent,
    RealtimeSessionOptions,
)


class FakeRealtimeSession:
    def __init__(self, events: Iterable[RealtimeEvent]) -> None:
        self._events = tuple(events)
        self.received_audio: list[bytes] = []
        self.close_count = 0

    async def send_audio_chunk(self, audio: bytes) -> None:
        self.received_audio.append(audio)

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        for event in self._events:
            yield event

    async def close(self) -> None:
        if self.close_count == 0:
            self.close_count = 1


class FakeRealtimeProvider:
    def __init__(self, events: Iterable[RealtimeEvent]) -> None:
        self._events = tuple(events)
        self.policy: DialoguePolicy | None = None
        self.options: RealtimeSessionOptions | None = None
        self.session: FakeRealtimeSession | None = None

    async def connect(
        self,
        policy: DialoguePolicy,
        options: RealtimeSessionOptions,
    ) -> FakeRealtimeSession:
        self.policy = policy
        self.options = options
        self.session = FakeRealtimeSession(self._events)
        return self.session
