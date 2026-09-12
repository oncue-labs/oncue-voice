import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from oncue_voice.session.results import CallOutcome, CallResult, CallStatus


class CallTerminationReason(str, Enum):
    USER_HANGUP = "USER_HANGUP"
    ABNORMAL_DISCONNECT = "ABNORMAL_DISCONNECT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    SCENARIO_COMPLETED = "SCENARIO_COMPLETED"
    TIME_LIMIT = "TIME_LIMIT"


class CallResultCallback(Protocol):
    def send_result(self, call_session_id: str, result: CallResult) -> None:
        """Send one final result to the backend."""


class CallLifecycle:
    """Track one call's internal termination and report it at most once."""

    def __init__(
        self,
        *,
        call_session_id: str,
        voice_session_id: str,
        result_callback: CallResultCallback | None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._call_session_id = call_session_id
        self._voice_session_id = voice_session_id
        self._result_callback = result_callback
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._call_status = CallStatus.CONNECTING
        self._started_at: datetime | None = None
        self._finished = False
        self._lock = asyncio.Lock()

    def mark_in_call(self) -> None:
        if self._finished or self._started_at is not None:
            return
        self._call_status = CallStatus.IN_CALL
        self._started_at = self._utc(self._clock())

    async def finish(self, reason: CallTerminationReason) -> CallResult | None:
        async with self._lock:
            if self._finished:
                return None

            result = CallResult(
                voiceSessionId=self._voice_session_id,
                callStatus=self._call_status,
                callOutcome=self._outcome(reason),
                startedAt=self._started_at,
                endedAt=self._utc(self._clock()),
            )
            if self._result_callback is not None:
                await asyncio.to_thread(
                    self._result_callback.send_result,
                    self._call_session_id,
                    result,
                )
            self._finished = True
            return result

    @staticmethod
    def _outcome(reason: CallTerminationReason) -> CallOutcome:
        if reason in (
            CallTerminationReason.USER_HANGUP,
            CallTerminationReason.SCENARIO_COMPLETED,
        ):
            return CallOutcome.SUCCEEDED
        return CallOutcome.FAILED

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
