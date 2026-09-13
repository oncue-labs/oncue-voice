import os
from collections.abc import Callable
from typing import Any

import redis
import uvicorn
from fastapi import FastAPI

from oncue_voice.api.http import create_internal_router
from oncue_voice.api.results import VoiceCallResultCallbackClient
from oncue_voice.api.signaling import (
    DefaultSignalingSessionFactory,
    RuntimeRunnerFactory,
    SignalingSessionFactory,
    create_signaling_router,
)
from oncue_voice.config import ServiceSettings
from oncue_voice.conversation.realtime_runtime import RealtimeRuntime
from oncue_voice.providers.realtime.factory import create_openai_realtime_provider
from oncue_voice.providers.realtime.models import RealtimeSessionOptions
from oncue_voice.session.auth import ConnectionTokenVerifier
from oncue_voice.session.jwt_config import JwtPublicKeySettings
from oncue_voice.session.lifecycle import CallResultCallback
from oncue_voice.session.service import SessionService
from oncue_voice.session.store import (
    InMemoryJtiStore,
    InMemoryVoiceSessionStore,
    JtiStore,
    RedisJtiStore,
    RedisVoiceSessionStore,
    VoiceSessionStore,
)


def create_app(
    *,
    session_service: SessionService | None = None,
    internal_service_token: str | None = None,
    voice_session_store: VoiceSessionStore | None = None,
    jti_store: JtiStore | None = None,
    token_verifier: ConnectionTokenVerifier | None = None,
    signaling_session_factory: SignalingSessionFactory | None = None,
    runtime_runner_factory: RuntimeRunnerFactory | None = None,
    result_callback: CallResultCallback | None = None,
    redis_client: Any | None = None,
) -> FastAPI:
    resolved_redis_client = _resolve_redis_client(
        redis_client,
        needs_store=voice_session_store is None or jti_store is None,
    )
    resolved_voice_session_store = voice_session_store or _create_voice_session_store(
        resolved_redis_client
    )
    resolved_jti_store = jti_store or _create_jti_store(resolved_redis_client)
    app = FastAPI(title="OnCue Voice")
    app.state.redis_client = resolved_redis_client
    app.state.voice_session_store = resolved_voice_session_store
    app.state.jti_store = resolved_jti_store
    app.include_router(
        create_internal_router(
            session_service or SessionService(resolved_voice_session_store),
            internal_service_token,
        )
    )

    if _signaling_is_requested(
        token_verifier=token_verifier,
        signaling_session_factory=signaling_session_factory,
        runtime_runner_factory=runtime_runner_factory,
    ):
        resolved_token_verifier = token_verifier or _create_token_verifier(
            resolved_jti_store
        )
        resolved_session_factory = signaling_session_factory
        if resolved_session_factory is None:
            resolved_runtime_runner_factory = (
                runtime_runner_factory or _create_runtime_runner_factory()
            )
            resolved_result_callback = result_callback or _create_result_callback(
                internal_service_token
            )
            resolved_session_factory = DefaultSignalingSessionFactory(
                resolved_voice_session_store,
                resolved_runtime_runner_factory,
                result_callback=resolved_result_callback,
            )
        app.include_router(
            create_signaling_router(
                resolved_token_verifier,
                resolved_session_factory,
            )
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _signaling_is_requested(
    *,
    token_verifier: ConnectionTokenVerifier | None,
    signaling_session_factory: SignalingSessionFactory | None,
    runtime_runner_factory: RuntimeRunnerFactory | None,
) -> bool:
    return any(
        dependency is not None
        for dependency in (
            token_verifier,
            signaling_session_factory,
            runtime_runner_factory,
        )
    ) or bool(
        os.getenv("ONCUE_VOICE_JWT_PUBLIC_KEY")
        or os.getenv("ONCUE_VOICE_JWT_PUBLIC_KEY_FILE")
    )


def _create_token_verifier(jti_store: JtiStore | None) -> ConnectionTokenVerifier:
    public_key_settings = JwtPublicKeySettings.from_environment()
    return ConnectionTokenVerifier(
        public_key_settings.public_key,
        jti_store or InMemoryJtiStore(),
    )


def _resolve_redis_client(
    redis_client: Any | None,
    *,
    needs_store: bool,
) -> Any | None:
    if redis_client is not None:
        return redis_client
    redis_host = os.getenv("REDIS_HOST", "").strip()
    if not redis_host or not needs_store:
        return None
    try:
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
    except ValueError as error:
        raise ValueError("REDIS_PORT must be an integer") from error
    return redis.Redis(host=redis_host, port=redis_port, decode_responses=False)


def _create_voice_session_store(redis_client: Any | None) -> VoiceSessionStore:
    if redis_client is None:
        return InMemoryVoiceSessionStore()
    return RedisVoiceSessionStore(redis_client)


def _create_jti_store(redis_client: Any | None) -> JtiStore:
    if redis_client is None:
        return InMemoryJtiStore()
    return RedisJtiStore(redis_client)


def _create_runtime_runner_factory() -> RuntimeRunnerFactory:
    runtime_name = os.getenv("ONCUE_VOICE_RUNTIME", "realtime").strip().lower()
    if runtime_name != "realtime":
        raise ValueError(
            "ONCUE_VOICE_RUNTIME must be 'realtime' until split_pipeline "
            "application wiring is implemented"
        )

    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime").strip()
    if not model:
        raise ValueError("OPENAI_REALTIME_MODEL must not be empty")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is required for realtime")

    def create_runner(session):
        async def run(audio):
            # Provider construction is inside the runner so app startup never
            # calls OpenAI. The first provider connection happens after offer.
            provider = create_openai_realtime_provider()
            runtime = RealtimeRuntime(provider)
            options = RealtimeSessionOptions(
                model=model,
                voice_id=session.policy_snapshot.voice_id,
            )
            async for event in runtime.run(
                session.voice_session_id,
                session.policy_snapshot,
                audio,
                options,
            ):
                yield event

        return run

    return create_runner


def _create_result_callback(
    internal_service_token: str | None,
) -> VoiceCallResultCallbackClient:
    backend_url = os.getenv("BACKEND_INTERNAL_URL", "").strip()
    if not backend_url:
        raise ValueError("BACKEND_INTERNAL_URL is required for signaling")

    service_token = (
        internal_service_token
        or os.getenv("ONCUE_VOICE_INTERNAL_SERVICE_TOKEN", "")
    ).strip()
    if not service_token:
        raise ValueError(
            "ONCUE_VOICE_INTERNAL_SERVICE_TOKEN is required for signaling"
        )
    return VoiceCallResultCallbackClient(backend_url, service_token)


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
