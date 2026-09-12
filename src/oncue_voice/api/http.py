import os
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from oncue_voice.api.schemas import (
    VoiceSessionCreatedResponse,
    VoiceSessionTerminatedResponse,
)
from oncue_voice.session.models import CreateSessionRequest
from oncue_voice.session.service import (
    InvalidSessionError,
    SessionConflictError,
    SessionNotFoundError,
    SessionService,
)


INTERNAL_SERVICE_TOKEN_ENV = "ONCUE_VOICE_INTERNAL_SERVICE_TOKEN"


def create_internal_router(
    session_service: SessionService,
    internal_service_token: str | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/internal/v1")
    expected_token = internal_service_token or os.getenv(
        INTERNAL_SERVICE_TOKEN_ENV
    )

    def require_service_auth(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if expected_token is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="internal service authentication is not configured",
            )
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="service authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        received_token = authorization.removeprefix("Bearer ")
        if not received_token or not secrets.compare_digest(
            received_token, expected_token
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid service authentication",
                headers={"WWW-Authenticate": "Bearer"},
            )

    service_auth = Depends(require_service_auth)

    @router.post(
        "/voice-sessions",
        response_model=VoiceSessionCreatedResponse,
        dependencies=[service_auth],
    )
    async def create_voice_session(
        request: CreateSessionRequest,
    ) -> VoiceSessionCreatedResponse:
        try:
            session = session_service.create(request)
        except SessionConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error
        except InvalidSessionError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error

        return VoiceSessionCreatedResponse(
            callSessionId=session.call_session_id,
            voiceSessionId=session.voice_session_id,
            createdAt=session.created_at,
        )

    @router.post(
        "/voice-sessions/{voice_session_id}/terminate",
        response_model=VoiceSessionTerminatedResponse,
        dependencies=[service_auth],
    )
    async def terminate_voice_session(
        voice_session_id: str,
    ) -> VoiceSessionTerminatedResponse:
        try:
            session_service.close(voice_session_id, reason="backend_terminate")
            session = session_service.get(voice_session_id)
        except SessionNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        return VoiceSessionTerminatedResponse(
            voiceSessionId=session.voice_session_id,
            status=session.status,
            closedAt=session.closed_at,
        )

    return router
