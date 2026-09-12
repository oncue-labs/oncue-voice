from collections.abc import AsyncIterator, Iterable, Mapping


class FakeTtsProvider:
    def __init__(
        self,
        audio_by_text: Mapping[str, Iterable[bytes]],
        events: list[str] | None = None,
    ) -> None:
        self._audio_by_text = {
            text: tuple(chunks) for text, chunks in audio_by_text.items()
        }
        self._events = events
        self.received_texts: list[str] = []

    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        if self._events is not None:
            self._events.append("tts")
        self.received_texts.append(text)
        for chunk in self._audio_by_text.get(text, ()):
            yield chunk
