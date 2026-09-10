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
