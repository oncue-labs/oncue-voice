from collections.abc import AsyncIterator
from typing import Any, Protocol


class SttProvider(Protocol):
    def stream_transcribe(
        self,
        audio: AsyncIterator[bytes],
    ) -> AsyncIterator[Any]:
        """Stream transcript segments from audio bytes."""
