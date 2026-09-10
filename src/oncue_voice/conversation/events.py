from typing import Literal

from pydantic import BaseModel


class ConversationEvent(BaseModel):
    type: Literal["user_turn", "assistant_chunk"]
    text: str
    sequence: int | None = None
