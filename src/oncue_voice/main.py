from collections.abc import Callable

import uvicorn
from fastapi import FastAPI

from oncue_voice.api.http import create_internal_router
from oncue_voice.config import ServiceSettings
from oncue_voice.session.service import SessionService
from oncue_voice.session.store import InMemoryVoiceSessionStore


def create_app(
    *,
    session_service: SessionService | None = None,
    internal_service_token: str | None = None,
) -> FastAPI:
    app = FastAPI(title="OnCue Voice")
    app.include_router(
        create_internal_router(
            session_service or SessionService(InMemoryVoiceSessionStore()),
            internal_service_token,
        )
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def run(
    server_runner: Callable[..., None] = uvicorn.run,
) -> None:
    settings = ServiceSettings.from_environment()
    server_runner(
        "oncue_voice.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )
