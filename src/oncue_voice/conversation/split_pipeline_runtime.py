import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from oncue_voice.conversation.events import ConversationEvent
from oncue_voice.conversation.models import (
    DialoguePolicy,
    TranscriptSegment,
    UserTurn,
)
from oncue_voice.providers.split_pipeline.models import ProviderBundle


MAX_CALL_DURATION_SECONDS = 300.0


@dataclass(frozen=True)
class SplitPipelineRuntimeOptions:
    clock: Callable[[], float] = time.monotonic
    max_duration_seconds: float = MAX_CALL_DURATION_SECONDS
    event_sink: Callable[[ConversationEvent], None] | None = None


class SplitPipelineRuntime:
    def __init__(
        self,
        providers: ProviderBundle,
        options: SplitPipelineRuntimeOptions | None = None,
    ) -> None:
        runtime_options = options or SplitPipelineRuntimeOptions()
        self._providers = providers
        self._clock = runtime_options.clock
        self._max_duration_seconds = runtime_options.max_duration_seconds
        self._event_sink = runtime_options.event_sink
        self._stopped_sessions: set[str] = set()

    async def run(
        self,
        session_id: str,
        policy: DialoguePolicy,
        audio: AsyncIterator[bytes],
    ) -> AsyncIterator[bytes]:
        if session_id in self._stopped_sessions:
            return

        started_at = self._clock()
        transcript_segments = [
            segment
            async for segment in self._providers.stt.stream_transcribe(audio)
        ]
        if self._contains_forbidden_topic(transcript_segments, policy):
            return

        async def user_turns() -> AsyncIterator[UserTurn]:
            for segment in transcript_segments:
                if segment.is_final:
                    self._emit_event(
                        ConversationEvent(type="user_turn", text=segment.text)
                    )
                    yield UserTurn(text=segment.text)

        async for assistant_chunk in self._providers.llm.stream_reply(
            policy,
            user_turns(),
        ):
            if self._has_stopped(session_id) or self._time_limit_reached(started_at):
                return
            self._emit_event(
                ConversationEvent(
                    type="assistant_chunk",
                    text=assistant_chunk.text,
                    sequence=assistant_chunk.sequence,
                )
            )
            async for audio_chunk in self._providers.tts.stream_synthesize(
                assistant_chunk.text
            ):
                if self._has_stopped(session_id) or self._time_limit_reached(
                    started_at
                ):
                    return
                yield audio_chunk

    def stop(self, session_id: str, reason: str) -> None:
        self._stopped_sessions.add(session_id)

    def _has_stopped(self, session_id: str) -> bool:
        return session_id in self._stopped_sessions

    def _time_limit_reached(self, started_at: float) -> bool:
        return self._clock() - started_at >= self._max_duration_seconds

    def _emit_event(self, event: ConversationEvent) -> None:
        if self._event_sink is not None:
            self._event_sink(event)

    @staticmethod
    def _contains_forbidden_topic(
        transcript_segments: list[TranscriptSegment],
        policy: DialoguePolicy,
    ) -> bool:
        normalized_text = " ".join(
            segment.text.casefold()
            for segment in transcript_segments
            if segment.is_final
        )
        return any(
            topic.casefold() in normalized_text for topic in policy.forbidden_topics
        )
