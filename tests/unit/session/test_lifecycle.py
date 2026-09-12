from datetime import datetime, timezone

import pytest

from oncue_voice.session.lifecycle import (
    CallLifecycle,
    CallTerminationReason,
)
from oncue_voice.session.results import CallOutcome, CallStatus


class FakeResultCallback:
    def __init__(self) -> None:
        self.calls = []

    def send_result(self, call_session_id, result) -> None:
        self.calls.append((call_session_id, result))


def create_clock():
    values = iter(
        (
            datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 12, 12, 5, tzinfo=timezone.utc),
        )
    )
    return lambda: next(values)


@pytest.mark.anyio
async def test_user_hangup_reports_successful_in_call_result_once() -> None:
    callback = FakeResultCallback()
    lifecycle = CallLifecycle(
        call_session_id="call-1",
        voice_session_id="voice-1",
        result_callback=callback,
        clock=create_clock(),
    )
    lifecycle.mark_in_call()

    first = await lifecycle.finish(CallTerminationReason.USER_HANGUP)
    second = await lifecycle.finish(CallTerminationReason.USER_HANGUP)

    assert first is not None
    assert first.call_status is CallStatus.IN_CALL
    assert first.call_outcome is CallOutcome.SUCCEEDED
    assert first.started_at == datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    assert first.ended_at == datetime(2026, 9, 12, 12, 5, tzinfo=timezone.utc)
    assert second is None
    assert len(callback.calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "reason",
    [
        CallTerminationReason.ABNORMAL_DISCONNECT,
        CallTerminationReason.PROVIDER_ERROR,
        CallTerminationReason.TIME_LIMIT,
    ],
)
async def test_technical_termination_reports_failed_result(
    reason: CallTerminationReason,
) -> None:
    callback = FakeResultCallback()
    lifecycle = CallLifecycle(
        call_session_id="call-1",
        voice_session_id="voice-1",
        result_callback=callback,
        clock=create_clock(),
    )
    lifecycle.mark_in_call()

    result = await lifecycle.finish(reason)

    assert result is not None
    assert result.call_status is CallStatus.IN_CALL
    assert result.call_outcome is CallOutcome.FAILED
    assert len(callback.calls) == 1


@pytest.mark.anyio
async def test_scenario_completion_reports_success_without_extra_contract_field() -> None:
    callback = FakeResultCallback()
    lifecycle = CallLifecycle(
        call_session_id="call-1",
        voice_session_id="voice-1",
        result_callback=callback,
        clock=create_clock(),
    )
    lifecycle.mark_in_call()

    result = await lifecycle.finish(CallTerminationReason.SCENARIO_COMPLETED)

    assert result is not None
    assert result.model_dump(by_alias=True, mode="json") == {
        "voiceSessionId": "voice-1",
        "callStatus": "IN_CALL",
        "callOutcome": "SUCCEEDED",
        "startedAt": "2026-09-12T12:00:00Z",
        "endedAt": "2026-09-12T12:05:00Z",
    }
