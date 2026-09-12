from collections.abc import AsyncIterator, Iterable

from oncue_voice.conversation.models import TranscriptSegment


class FakeSttProvider:
    def __init__(
        self,
        segments: Iterable[TranscriptSegment | dict[str, object]],
        events: list[str] | None = None,
    ) -> None:
        self._segments = tuple(
            segment
            if isinstance(segment, TranscriptSegment)
            else TranscriptSegment.model_validate(segment)
            for segment in segments
        )
        self._events = events

    async def stream_transcribe(
        self,
        audio: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptSegment]:
        if self._events is not None:
            self._events.append("stt")
        async for _ in audio:
            pass
        for segment in self._segments:
            yield segment
