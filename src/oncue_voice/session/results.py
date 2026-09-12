from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CallStatus(str, Enum):
    """통화가 끝나기 직전까지 진행된 마지막 단계."""

    PREPARING = "PREPARING"
    RINGING = "RINGING"
    CONNECTING = "CONNECTING"
    IN_CALL = "IN_CALL"


class CallOutcome(str, Enum):
    """통화가 성공 또는 실패로 종료되었는지 나타내는 값."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class CallResult(BaseModel):
    """보이스 서버가 백엔드에 한 번 전달하는 최종 통화 결과."""

    model_config = ConfigDict(populate_by_name=True)

    voice_session_id: str = Field(alias="voiceSessionId")
    call_status: CallStatus = Field(alias="callStatus")
    call_outcome: CallOutcome = Field(alias="callOutcome")
    started_at: datetime | None = Field(default=None, alias="startedAt")
    ended_at: datetime = Field(alias="endedAt")
