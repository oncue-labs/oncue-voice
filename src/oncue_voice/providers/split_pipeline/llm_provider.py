from collections.abc import AsyncIterator
from typing import Any, Protocol


class LlmProvider(Protocol):
    def stream_reply(
        self,
        policy: Any,
        turns: AsyncIterator[Any],
    ) -> AsyncIterator[Any]:
        """Stream assistant chunks for the supplied dialogue turns."""
