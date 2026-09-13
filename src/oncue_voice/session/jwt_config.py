import os
from dataclasses import dataclass
from pathlib import Path


PUBLIC_KEY_ENV_NAME = "ONCUE_VOICE_JWT_PUBLIC_KEY"
PUBLIC_KEY_FILE_ENV_NAME = "ONCUE_VOICE_JWT_PUBLIC_KEY_FILE"


@dataclass(frozen=True)
class JwtPublicKeySettings:
    """통화 연결 토큰 서명 검증에 사용할 RSA 공개키 설정."""

    # 백엔드가 서명한 통화 연결 토큰을 검증할 공개키
    public_key: str

    @classmethod
    def from_environment(cls) -> "JwtPublicKeySettings":
        public_key = _read_environment_value(
            PUBLIC_KEY_ENV_NAME,
            PUBLIC_KEY_FILE_ENV_NAME,
        )
        if not public_key:
            raise ValueError(
                f"{PUBLIC_KEY_ENV_NAME} or {PUBLIC_KEY_FILE_ENV_NAME} is required"
            )
        return cls(public_key=public_key)


def _read_environment_value(value_name: str, file_name: str) -> str:
    value = os.getenv(value_name)
    if value and value.strip():
        return _normalize_pem(value)

    path = os.getenv(file_name)
    if not path:
        return ""
    try:
        return _normalize_pem(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to read {file_name}") from error


def _normalize_pem(value: str) -> str:
    return value.replace("\\n", "\n").strip()
