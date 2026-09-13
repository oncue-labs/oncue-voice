from datetime import datetime, timezone

import httpx
import pytest

from oncue_voice.api.results import VoiceCallResultCallbackClient
from oncue_voice.main import create_app
from oncue_voice.session.results import CallResult
from oncue_voice.session.service import SessionService
from oncue_voice.session.store import InMemoryVoiceSessionStore


SERVICE_TOKEN = "backend-service-token"


def policy_payload() -> dict[str, object]:
    return {
        "role": "Santa",
        "stages": ["greeting", "goal"],
        "goal": "help the child get ready for bed",
        "language": "ko-KR",
        "voiceId": "alloy",
        "scenarioContext": "The child is preparing for bed.",
    }


def session_payload(
    *,
    call_session_id: int = 1,
    user_id: int = 7,
) -> dict[str, object]:
    return {
        "callSessionId": call_session_id,
        "userId": user_id,
        "policySnapshot": policy_payload(),
        "expiresAt": "2099-01-01T00:00:00Z",
    }


def auth_headers(token: str = SERVICE_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def create_client() -> tuple[httpx.AsyncClient, SessionService]:
    service = SessionService(InMemoryVoiceSessionStore())
    app = create_app(
        session_service=service,
        internal_service_token=SERVICE_TOKEN,
    )
    transport = httpx.ASGITransport(app=app)
    return (
        httpx.AsyncClient(transport=transport, base_url="http://test"),
        service,
    )


@pytest.mark.anyio
async def test_create_voice_session_requires_internal_service_token() -> None:
    client, _ = create_client()
    async with client:
        response = await client.post(
            "/internal/v1/voice-sessions",
            json=session_payload(),
            headers={"Authorization": "Bearer mobile-access-token"},
        )

    assert response.status_code == 401


@pytest.mark.anyio
async def test_create_voice_session_returns_internal_id_and_created_at() -> None:
    client, service = create_client()
    async with client:
        response = await client.post(
            "/internal/v1/voice-sessions",
            json=session_payload(),
            headers=auth_headers(),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["callSessionId"] == 1
    assert body["voiceSessionId"]
    assert body["createdAt"]
    assert service.get(body["voiceSessionId"]).call_session_id == 1


@pytest.mark.anyio
async def test_create_voice_session_requires_dialogue_policy_fields() -> None:
    client, _ = create_client()
    payload = session_payload()
    payload["policySnapshot"] = {"role": "Santa"}

    async with client:
        response = await client.post(
            "/internal/v1/voice-sessions",
            json=payload,
            headers=auth_headers(),
        )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_create_voice_session_is_idempotent_for_same_call_session() -> None:
    client, _ = create_client()
    async with client:
        first = await client.post(
            "/internal/v1/voice-sessions",
            json=session_payload(),
            headers=auth_headers(),
        )
        second = await client.post(
            "/internal/v1/voice-sessions",
            json=session_payload(),
            headers=auth_headers(),
        )

    assert first.status_code == second.status_code == 200
    assert second.json()["voiceSessionId"] == first.json()["voiceSessionId"]


@pytest.mark.anyio
async def test_create_voice_session_rejects_conflicting_duplicate() -> None:
    client, _ = create_client()
    async with client:
        await client.post(
            "/internal/v1/voice-sessions",
            json=session_payload(),
            headers=auth_headers(),
        )
        response = await client.post(
            "/internal/v1/voice-sessions",
            json=session_payload(user_id=8),
            headers=auth_headers(),
        )

    assert response.status_code == 409


@pytest.mark.anyio
async def test_terminate_voice_session_closes_session_and_is_idempotent() -> None:
    client, _ = create_client()
    async with client:
        created = await client.post(
            "/internal/v1/voice-sessions",
            json=session_payload(),
            headers=auth_headers(),
        )
        voice_session_id = created.json()["voiceSessionId"]
        first = await client.post(
            f"/internal/v1/voice-sessions/{voice_session_id}/terminate",
            headers=auth_headers(),
        )
        second = await client.post(
            f"/internal/v1/voice-sessions/{voice_session_id}/terminate",
            headers=auth_headers(),
        )

    assert first.status_code == second.status_code == 200
    assert first.json()["status"] == second.json()["status"] == "CLOSED"


@pytest.mark.anyio
async def test_terminate_unknown_voice_session_returns_not_found() -> None:
    client, _ = create_client()
    async with client:
        response = await client.post(
            "/internal/v1/voice-sessions/unknown/terminate",
            headers=auth_headers(),
        )

    assert response.status_code == 404


def test_call_result_uses_contract_field_names() -> None:
    result = CallResult(
        voiceSessionId="voice-1",
        callStatus="IN_CALL",
        callOutcome="SUCCEEDED",
        startedAt=datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
        endedAt=datetime(2026, 9, 12, 12, 5, tzinfo=timezone.utc),
    )

    assert result.model_dump(by_alias=True, mode="json") == {
        "voiceSessionId": "voice-1",
        "callStatus": "IN_CALL",
        "callOutcome": "SUCCEEDED",
        "startedAt": "2026-09-12T12:00:00Z",
        "endedAt": "2026-09-12T12:05:00Z",
    }


class FakeResponse:
    def raise_for_status(self) -> None:
        return None


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse()


def test_result_callback_sends_service_authenticated_contract_payload() -> None:
    client = FakeHttpClient()
    callback = VoiceCallResultCallbackClient(
        base_url="http://backend",
        service_token=SERVICE_TOKEN,
        http_client=client,
    )
    result = CallResult(
        voiceSessionId="voice-1",
        callStatus="IN_CALL",
        callOutcome="SUCCEEDED",
        startedAt=datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
        endedAt=datetime(2026, 9, 12, 12, 5, tzinfo=timezone.utc),
    )

    callback.send_result(1, result)

    assert client.calls == [
        {
            "url": "/internal/v1/call-sessions/1/result",
            "json": result.model_dump(by_alias=True, mode="json"),
            "headers": {"Authorization": f"Bearer {SERVICE_TOKEN}"},
        }
    ]
