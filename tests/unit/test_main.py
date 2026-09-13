import pytest

from oncue_voice.main import create_app
from oncue_voice.session.store import InMemoryVoiceSessionStore


def test_create_app_registers_signaling_route_from_public_key_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONCUE_VOICE_JWT_PUBLIC_KEY", "test-public-key")
    app = create_app(
        signaling_session_factory=object(),
        voice_session_store=InMemoryVoiceSessionStore(),
    )

    assert any(
        getattr(child, "path", None)
        == "/v1/signaling/call-sessions/{call_session_id}"
        for router in app.routes
        for child in getattr(getattr(router, "original_router", None), "routes", ())
    )


def test_create_app_requires_public_key_when_signaling_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONCUE_VOICE_JWT_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("ONCUE_VOICE_JWT_PUBLIC_KEY_FILE", raising=False)

    with pytest.raises(ValueError, match="ONCUE_VOICE_JWT_PUBLIC_KEY"):
        create_app(
            signaling_session_factory=object(),
            voice_session_store=InMemoryVoiceSessionStore(),
        )


def test_create_app_builds_default_realtime_wiring_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONCUE_VOICE_JWT_PUBLIC_KEY", "test-public-key")
    monkeypatch.setenv("BACKEND_INTERNAL_URL", "http://backend:8080")
    monkeypatch.setenv("ONCUE_VOICE_INTERNAL_SERVICE_TOKEN", "service-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    app = create_app(voice_session_store=InMemoryVoiceSessionStore())

    assert any(
        getattr(child, "path", None)
        == "/v1/signaling/call-sessions/{call_session_id}"
        for router in app.routes
        for child in getattr(getattr(router, "original_router", None), "routes", ())
    )


def test_create_app_requires_backend_callback_configuration_for_default_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONCUE_VOICE_JWT_PUBLIC_KEY", "test-public-key")
    monkeypatch.delenv("BACKEND_INTERNAL_URL", raising=False)
    monkeypatch.delenv("ONCUE_VOICE_INTERNAL_SERVICE_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    with pytest.raises(ValueError, match="BACKEND_INTERNAL_URL"):
        create_app(voice_session_store=InMemoryVoiceSessionStore())
