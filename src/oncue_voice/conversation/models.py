from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DialoguePolicy(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: str
    stages: tuple[str, ...]
    goal: str
    allowed_topics: tuple[str, ...] = Field(
        default=(),
        alias="allowedTopics",
    )
    forbidden_topics: tuple[str, ...] = Field(
        default=(),
        alias="forbiddenTopics",
    )
    termination_conditions: tuple[str, ...] = Field(
        default=(),
        alias="terminationConditions",
    )
    language: str
    voice_id: str = Field(alias="voiceId")
    instructions: tuple[str, ...] = ()
    dialogue_rules: tuple[str, ...] = Field(
        default=(),
        alias="dialogueRules",
    )
    scenario_context: str = Field(alias="scenarioContext")
    voice_settings: dict[str, Any] = Field(
        default_factory=dict,
        alias="voiceSettings",
    )


class UserTurn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        alias="createdAt",
    )


class AssistantChunk(BaseModel):
    text: str
    sequence: int


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    is_final: bool = Field(alias="isFinal")
    start_ms: int = Field(alias="startMs")
    end_ms: int = Field(alias="endMs")
