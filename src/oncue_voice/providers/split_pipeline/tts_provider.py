from collections.abc import AsyncIterator
from typing import Protocol


class TtsProvider(Protocol):
    def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesized audio for text."""
