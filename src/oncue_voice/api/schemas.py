from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from oncue_voice.session.models import SessionStatus


class VoiceSessionCreatedResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # 생성된 보이스 세션과 연결된 백엔드 통화 세션을 식별하는 값이다.
    call_session_id: int = Field(alias="callSessionId", strict=True)
    voice_session_id: str = Field(alias="voiceSessionId")
    created_at: datetime = Field(alias="createdAt")


class VoiceSessionTerminatedResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    voice_session_id: str = Field(alias="voiceSessionId")
    status: SessionStatus
    closed_at: datetime | None = Field(alias="closedAt")
