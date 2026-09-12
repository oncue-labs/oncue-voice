import asyncio
from collections.abc import AsyncIterator

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.providers.realtime.models import (
    RealtimeEvent,
    RealtimeSessionOptions,
)
from oncue_voice.providers.realtime.provider import RealtimeProvider, RealtimeSession


class RealtimeRuntime:
    def __init__(self, provider: RealtimeProvider) -> None:
        self._provider = provider

    async def run(
        self,
        session_id: str,
        policy: DialoguePolicy,
        audio: AsyncIterator[bytes],
        options: RealtimeSessionOptions,
    ) -> AsyncIterator[RealtimeEvent]:
        del session_id
        session = await self._provider.connect(policy, options)
        send_task = asyncio.create_task(self._send_audio(session, audio))
        events_completed = False
        try:
            async for event in session.events():
                yield event
            events_completed = True
        finally:
            await self._finish_audio_sender(send_task, events_completed)
            await session.close()

    @staticmethod
    async def _send_audio(
        session: RealtimeSession,
        audio: AsyncIterator[bytes],
    ) -> None:
        async for chunk in audio:
            await session.send_audio_chunk(chunk)

    @staticmethod
    async def _finish_audio_sender(
        send_task: asyncio.Task[None],
        events_completed: bool,
    ) -> None:
        if events_completed:
            await send_task
            return

        if not send_task.done():
            send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            return
