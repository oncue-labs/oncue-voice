from collections.abc import Callable

import uvicorn
from fastapi import FastAPI

from oncue_voice.config import ServiceSettings


def create_app() -> FastAPI:
    app = FastAPI(title="OnCue Voice")

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
