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

    call_session_id: str = Field(alias="callSessionId")
    user_id: str = Field(alias="userId")
    scope: tuple[str, ...]
    jti: str
    iat: int
    exp: int


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    call_session_id: str = Field(alias="callSessionId")
    user_id: str = Field(alias="userId")
    policy: DialoguePolicy
    expires_at: datetime = Field(alias="expiresAt")


class VoiceSession(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    voice_session_id: str = Field(alias="voiceSessionId")
    call_session_id: str = Field(alias="callSessionId")
    user_id: str = Field(alias="userId")
    policy: DialoguePolicy
    expires_at: datetime = Field(alias="expiresAt")
    status: SessionStatus
    created_at: datetime = Field(alias="createdAt")
    closed_at: datetime | None = Field(default=None, alias="closedAt")

    def policy_snapshot(self) -> dict[str, Any]:
        return self.policy.model_dump(mode="json", by_alias=True)
