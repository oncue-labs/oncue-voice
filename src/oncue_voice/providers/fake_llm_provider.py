from collections.abc import AsyncIterator, Iterable
from typing import Any

from oncue_voice.conversation.models import AssistantChunk, DialoguePolicy, UserTurn


class FakeLlmProvider:
    def __init__(
        self,
        chunks: Iterable[AssistantChunk | dict[str, object]],
        events: list[str] | None = None,
    ) -> None:
        self._chunks = tuple(
            chunk
            if isinstance(chunk, AssistantChunk)
            else AssistantChunk.model_validate(chunk)
            for chunk in chunks
        )
        self._events = events
        self.last_policy: DialoguePolicy | None = None
        self.received_turns: list[UserTurn] = []

    async def stream_reply(
        self,
        policy: DialoguePolicy,
        turns: AsyncIterator[UserTurn],
    ) -> AsyncIterator[AssistantChunk]:
        if self._events is not None:
            self._events.append("llm")
        self.last_policy = policy
        self.received_turns = [turn async for turn in turns]
        for chunk in self._chunks:
            yield chunk
