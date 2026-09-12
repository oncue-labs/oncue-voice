from pathlib import Path

import pytest

from oncue_voice.config import TurnSettings


def test_turn_settings_read_urls_and_credentials_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ONCUE_VOICE_TURN_URLS",
        "turn:turn.example:3478?transport=udp, turns:turn.example:5349",
    )
    monkeypatch.setenv("ONCUE_VOICE_TURN_USERNAME", "temporary-user")
    monkeypatch.setenv("ONCUE_VOICE_TURN_CREDENTIAL", "temporary-credential")

    settings = TurnSettings.from_environment()

    assert settings.urls == (
        "turn:turn.example:3478?transport=udp",
        "turns:turn.example:5349",
    )
    assert settings.username == "temporary-user"
    assert settings.credential == "temporary-credential"


def test_turn_settings_read_credentials_from_docker_secret_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    username_file = tmp_path / "turn-username"
    credential_file = tmp_path / "turn-credential"
    username_file.write_text("temporary-user\n")
    credential_file.write_text("temporary-credential\n")
    monkeypatch.setenv("ONCUE_VOICE_TURN_URLS", "turn:coturn:3478")
    monkeypatch.setenv("ONCUE_VOICE_TURN_USERNAME_FILE", str(username_file))
    monkeypatch.setenv("ONCUE_VOICE_TURN_CREDENTIAL_FILE", str(credential_file))

    settings = TurnSettings.from_environment()

    assert settings.username == "temporary-user"
    assert settings.credential == "temporary-credential"


def test_turn_settings_require_urls_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONCUE_VOICE_TURN_URLS", raising=False)
    monkeypatch.delenv("ONCUE_VOICE_TURN_USERNAME", raising=False)
    monkeypatch.delenv("ONCUE_VOICE_TURN_CREDENTIAL", raising=False)
    monkeypatch.delenv("ONCUE_VOICE_TURN_USERNAME_FILE", raising=False)
    monkeypatch.delenv("ONCUE_VOICE_TURN_CREDENTIAL_FILE", raising=False)

    with pytest.raises(ValueError, match="ONCUE_VOICE_TURN_URLS"):
        TurnSettings.from_environment()
