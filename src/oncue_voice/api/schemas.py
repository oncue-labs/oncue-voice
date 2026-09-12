from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from oncue_voice.session.models import SessionStatus


class VoiceSessionCreatedResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    call_session_id: str = Field(alias="callSessionId")
    voice_session_id: str = Field(alias="voiceSessionId")
    created_at: datetime = Field(alias="createdAt")


class VoiceSessionTerminatedResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    voice_session_id: str = Field(alias="voiceSessionId")
    status: SessionStatus
    closed_at: datetime | None = Field(alias="closedAt")
