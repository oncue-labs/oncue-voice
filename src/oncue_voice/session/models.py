from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from oncue_voice.conversation.models import DialoguePolicy


class SessionStatus(str, Enum):
    PREPARED = "PREPARED"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"


class ConnectionClaims(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # 백엔드의 자동 증가 통화 세션을 식별하는 값이다.
    call_session_id: int = Field(alias="callSessionId", strict=True)
    # 통화를 예약한 백엔드 사용자를 식별하는 값이다.
    user_id: int = Field(alias="userId", strict=True)
    scope: tuple[str, ...]
    # 토큰을 한 번만 사용할 수 있게 추적하는 고유 식별자다.
    jti: str
    # 토큰이 발급된 시각이다. Unix 초 단위로 표현한다.
    iat: int
    # 토큰이 더 이상 유효하지 않은 만료 시각이다. Unix 초 단위다.
    exp: int


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # 준비할 백엔드 통화 세션을 식별하는 값이다.
    call_session_id: int = Field(alias="callSessionId", strict=True)
    # 준비한 통화를 사용할 백엔드 사용자를 식별하는 값이다.
    user_id: int = Field(alias="userId", strict=True)
    policy_snapshot: DialoguePolicy = Field(alias="policySnapshot")
    expires_at: datetime = Field(alias="expiresAt")


class VoiceSession(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    voice_session_id: str = Field(alias="voiceSessionId")
    # 이 보이스 세션에 연결된 백엔드 통화 세션을 식별하는 값이다.
    call_session_id: int = Field(alias="callSessionId", strict=True)
    # 이 보이스 세션으로 연결할 백엔드 사용자를 식별하는 값이다.
    user_id: int = Field(alias="userId", strict=True)
    policy_snapshot: DialoguePolicy = Field(alias="policySnapshot")
    expires_at: datetime = Field(alias="expiresAt")
    status: SessionStatus
    created_at: datetime = Field(alias="createdAt")
    closed_at: datetime | None = Field(default=None, alias="closedAt")

    def policy_snapshot_json(self) -> dict[str, Any]:
        return self.policy_snapshot.model_dump(mode="json", by_alias=True)
