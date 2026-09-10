import httpx
import pytest

from oncue_voice.main import create_app


@pytest.mark.anyio
async def test_health_endpoint_reports_service_is_ready() -> None:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
