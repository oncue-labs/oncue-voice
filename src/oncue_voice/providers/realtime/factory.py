import os
from typing import Any

from openai import AsyncOpenAI

from oncue_voice.providers.realtime.openai_provider import OpenAiRealtimeProvider


def create_openai_realtime_provider(
    client: Any | None = None,
    api_key: str | None = None,
) -> OpenAiRealtimeProvider:
    resolved_client = client
    if resolved_client is None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        resolved_client = AsyncOpenAI(api_key=resolved_api_key)
    return OpenAiRealtimeProvider(resolved_client)
