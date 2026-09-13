from pathlib import Path

import pytest

from oncue_voice.session.jwt_config import JwtPublicKeySettings


def test_reads_inline_public_key_and_expands_literal_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ONCUE_VOICE_JWT_PUBLIC_KEY",
        "-----BEGIN PUBLIC KEY-----\\npublic-key\\n-----END PUBLIC KEY-----",
    )

    settings = JwtPublicKeySettings.from_environment()

    assert settings.public_key == (
        "-----BEGIN PUBLIC KEY-----\npublic-key\n-----END PUBLIC KEY-----"
    )


def test_inline_public_key_takes_precedence_over_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    public_key_file = tmp_path / "jwt-public-key.pem"
    public_key_file.write_text("file-public-key", encoding="utf-8")
    monkeypatch.setenv("ONCUE_VOICE_JWT_PUBLIC_KEY", "inline-public-key")
    monkeypatch.setenv("ONCUE_VOICE_JWT_PUBLIC_KEY_FILE", str(public_key_file))

    settings = JwtPublicKeySettings.from_environment()

    assert settings.public_key == "inline-public-key"


def test_reads_public_key_from_file_when_inline_value_is_not_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    public_key_file = tmp_path / "jwt-public-key.pem"
    public_key_file.write_text(
        "-----BEGIN PUBLIC KEY-----\\nfile-key\\n-----END PUBLIC KEY-----\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ONCUE_VOICE_JWT_PUBLIC_KEY", raising=False)
    monkeypatch.setenv("ONCUE_VOICE_JWT_PUBLIC_KEY_FILE", str(public_key_file))

    settings = JwtPublicKeySettings.from_environment()

    assert settings.public_key == (
        "-----BEGIN PUBLIC KEY-----\nfile-key\n-----END PUBLIC KEY-----"
    )


def test_requires_inline_value_or_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ONCUE_VOICE_JWT_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("ONCUE_VOICE_JWT_PUBLIC_KEY_FILE", raising=False)

    with pytest.raises(ValueError, match="ONCUE_VOICE_JWT_PUBLIC_KEY"):
        JwtPublicKeySettings.from_environment()
