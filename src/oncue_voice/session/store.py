import math
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

from oncue_voice.session.models import VoiceSession


class JtiStore(Protocol):
    def consume(self, jti: str, ttl_seconds: int) -> bool:
        """Atomically mark a JTI as used for the token's remaining lifetime."""


class InMemoryJtiStore:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._expires_at_by_jti: dict[str, float] = {}

    def consume(self, jti: str, ttl_seconds: int) -> bool:
        now = self._clock()
        self._expires_at_by_jti = {
            stored_jti: expires_at
            for stored_jti, expires_at in self._expires_at_by_jti.items()
            if expires_at > now
        }
        if jti in self._expires_at_by_jti:
            return False
        self._expires_at_by_jti[jti] = now + max(ttl_seconds, 1)
        return True


class RedisJtiStore:
    def __init__(
        self,
        client: Any,
        key_prefix: str = "oncue:voice:connection-jti",
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix

    def consume(self, jti: str, ttl_seconds: int) -> bool:
        result = self._client.set(
            self._key(jti),
            "1",
            nx=True,
            ex=max(ttl_seconds, 1),
        )
        return bool(result)

    def _key(self, jti: str) -> str:
        return f"{self._key_prefix}:{jti}"


class VoiceSessionStore(Protocol):
    def get_by_call_session_id(self, call_session_id: int) -> VoiceSession | None:
        """Find a voice session by the public OnCue call session ID."""

    def get_by_voice_session_id(self, voice_session_id: str) -> VoiceSession | None:
        """Find a voice session by the voice service's internal ID."""

    def save(self, session: VoiceSession) -> None:
        """Create or replace short-lived technical session state."""


class InMemoryVoiceSessionStore:
    def __init__(self) -> None:
        self._sessions_by_voice_id: dict[str, VoiceSession] = {}
        self._voice_id_by_call_id: dict[int, str] = {}

    def get_by_call_session_id(self, call_session_id: int) -> VoiceSession | None:
        voice_session_id = self._voice_id_by_call_id.get(call_session_id)
        if voice_session_id is None:
            return None
        return self.get_by_voice_session_id(voice_session_id)

    def get_by_voice_session_id(self, voice_session_id: str) -> VoiceSession | None:
        return self._sessions_by_voice_id.get(voice_session_id)

    def save(self, session: VoiceSession) -> None:
        self._sessions_by_voice_id[session.voice_session_id] = session
        self._voice_id_by_call_id[session.call_session_id] = session.voice_session_id


class RedisVoiceSessionStore:
    def __init__(
        self,
        client: Any,
        key_prefix: str = "oncue:voice:session",
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix

    def get_by_call_session_id(self, call_session_id: int) -> VoiceSession | None:
        voice_session_id = self._decode(self._client.get(self._call_key(call_session_id)))
        if voice_session_id is None:
            return None
        return self.get_by_voice_session_id(voice_session_id)

    def get_by_voice_session_id(self, voice_session_id: str) -> VoiceSession | None:
        payload = self._client.get(self._voice_key(voice_session_id))
        if payload is None:
            return None
        return VoiceSession.model_validate_json(payload)

    def save(self, session: VoiceSession) -> None:
        ttl_seconds = self._ttl_seconds(session.expires_at)
        payload = session.model_dump_json(by_alias=True)
        self._client.set(
            self._voice_key(session.voice_session_id),
            payload,
            ex=ttl_seconds,
        )
        self._client.set(
            self._call_key(session.call_session_id),
            session.voice_session_id,
            ex=ttl_seconds,
        )

    def _voice_key(self, voice_session_id: str) -> str:
        return f"{self._key_prefix}:voice:{voice_session_id}"

    def _call_key(self, call_session_id: int) -> str:
        return f"{self._key_prefix}:call:{call_session_id}"

    @staticmethod
    def _ttl_seconds(expires_at: datetime) -> int:
        now = datetime.now(timezone.utc)
        remaining = (expires_at - now).total_seconds()
        return max(math.ceil(remaining), 1)

    @staticmethod
    def _decode(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)
