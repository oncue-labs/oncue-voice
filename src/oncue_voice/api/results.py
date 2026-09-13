from typing import Protocol

import httpx

from oncue_voice.session.results import CallResult


class _HttpResponse(Protocol):
    def raise_for_status(self) -> None:
        """Raise when the backend rejected the callback."""


class _HttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> _HttpResponse:
        """Send one JSON request to the backend."""


class VoiceCallResultCallbackClient:
    """Sends final call results from voice service to backend."""

    def __init__(
        self,
        base_url: str,
        service_token: str,
        http_client: _HttpClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must not be empty")
        if not service_token:
            raise ValueError("service_token must not be empty")
        self._http_client = http_client or httpx.Client(base_url=base_url)
        self._service_token = service_token

    def send_result(self, call_session_id: int, result: CallResult) -> None:
        response = self._http_client.post(
            f"/internal/v1/call-sessions/{call_session_id}/result",
            json=result.model_dump(by_alias=True, mode="json"),
            headers={"Authorization": f"Bearer {self._service_token}"},
        )
        response.raise_for_status()
