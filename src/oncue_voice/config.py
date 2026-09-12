import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_environment(cls) -> "ServiceSettings":
        return cls(
            host=os.getenv("ONCUE_VOICE_HOST", cls.host),
            port=int(os.getenv("ONCUE_VOICE_PORT", str(cls.port))),
        )


@dataclass(frozen=True)
class TurnSettings:
    """Coturn connection settings shared with the mobile ICE response."""

    urls: tuple[str, ...]
    username: str
    credential: str

    @classmethod
    def from_environment(cls) -> "TurnSettings":
        urls_value = os.getenv("ONCUE_VOICE_TURN_URLS", "")
        urls = tuple(url.strip() for url in urls_value.split(",") if url.strip())
        if not urls:
            raise ValueError("ONCUE_VOICE_TURN_URLS is required")

        username = _read_secret(
            "ONCUE_VOICE_TURN_USERNAME",
            "ONCUE_VOICE_TURN_USERNAME_FILE",
        )
        credential = _read_secret(
            "ONCUE_VOICE_TURN_CREDENTIAL",
            "ONCUE_VOICE_TURN_CREDENTIAL_FILE",
        )
        if not username:
            raise ValueError(
                "ONCUE_VOICE_TURN_USERNAME or "
                "ONCUE_VOICE_TURN_USERNAME_FILE is required"
            )
        if not credential:
            raise ValueError(
                "ONCUE_VOICE_TURN_CREDENTIAL or "
                "ONCUE_VOICE_TURN_CREDENTIAL_FILE is required"
            )
        return cls(urls=urls, username=username, credential=credential)


def _read_secret(value_name: str, file_name: str) -> str:
    value = os.getenv(value_name)
    if value:
        return value.strip()
    path = os.getenv(file_name)
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8") as secret_file:
            return secret_file.read().strip()
    except OSError as error:
        raise ValueError(f"unable to read {file_name}") from error
