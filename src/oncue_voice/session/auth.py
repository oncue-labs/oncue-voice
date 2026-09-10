import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jwt

from oncue_voice.session.models import ConnectionClaims
from oncue_voice.session.store import JtiStore


VOICE_CONNECT_SCOPE = "voice:connect"
REQUIRED_CLAIMS = ("callSessionId", "userId", "scope", "jti", "iat", "exp")


class InvalidConnectionTokenError(ValueError):
    """Raised when a mobile-to-voice connection token is not usable."""


@dataclass(frozen=True)
class ConnectionTokenVerifierOptions:
    algorithm: str = "RS256"
    required_scope: str = VOICE_CONNECT_SCOPE
    issuer: str | None = None
    audience: str | None = None
    leeway_seconds: int = 0
    clock: Callable[[], float] = time.time


class ConnectionTokenVerifier:
    def __init__(
        self,
        public_key: str,
        jti_store: JtiStore,
        options: ConnectionTokenVerifierOptions | None = None,
    ) -> None:
        self._public_key = public_key
        self._jti_store = jti_store
        self._options = options or ConnectionTokenVerifierOptions()

    def verify(
        self,
        token: str,
        call_session_id: str,
        user_id: str | None = None,
    ) -> ConnectionClaims:
        payload = self._decode(token)
        now = self._options.clock()
        call_session_claim = self._identifier(payload.get("callSessionId"))
        user_claim = self._identifier(payload.get("userId"))
        jti = self._non_empty_string(payload.get("jti"))
        scope = self._scope(payload.get("scope"))
        issued_at = self._integer(payload.get("iat"))
        expires_at = self._integer(payload.get("exp"))

        if call_session_claim != str(call_session_id):
            raise self._invalid()
        if user_id is not None and user_claim != str(user_id):
            raise self._invalid()
        if self._options.required_scope not in scope:
            raise self._invalid()
        if expires_at <= now - self._options.leeway_seconds:
            raise self._invalid()
        if issued_at > now + self._options.leeway_seconds:
            raise self._invalid()

        ttl_seconds = max(int(expires_at - now), 1)
        if not self._jti_store.consume(jti, ttl_seconds):
            raise self._invalid()

        return ConnectionClaims(
            callSessionId=call_session_claim,
            userId=user_claim,
            scope=scope,
            jti=jti,
            iat=issued_at,
            exp=expires_at,
        )

    def _decode(self, token: str) -> dict[str, Any]:
        options = {
            "require": REQUIRED_CLAIMS,
            "verify_exp": False,
            "verify_iat": False,
        }
        decode_kwargs: dict[str, Any] = {
            "algorithms": [self._options.algorithm],
            "options": options,
        }
        if self._options.issuer is not None:
            decode_kwargs["issuer"] = self._options.issuer
        else:
            options["verify_iss"] = False
        if self._options.audience is not None:
            decode_kwargs["audience"] = self._options.audience
        else:
            options["verify_aud"] = False

        try:
            payload = jwt.decode(token, self._public_key, **decode_kwargs)
        except (jwt.PyJWTError, TypeError, ValueError) as error:
            raise self._invalid() from error
        if not isinstance(payload, dict):
            raise self._invalid()
        return payload

    @staticmethod
    def _identifier(value: object) -> str:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise InvalidConnectionTokenError("invalid connection token")
        identifier = str(value)
        if not identifier:
            raise InvalidConnectionTokenError("invalid connection token")
        return identifier

    @staticmethod
    def _non_empty_string(value: object) -> str:
        if not isinstance(value, str) or not value:
            raise InvalidConnectionTokenError("invalid connection token")
        return value

    @staticmethod
    def _scope(value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            scopes = tuple(value.split())
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, str) and item for item in value
        ):
            scopes = tuple(value)
        else:
            raise InvalidConnectionTokenError("invalid connection token")
        if not scopes:
            raise InvalidConnectionTokenError("invalid connection token")
        return scopes

    @staticmethod
    def _integer(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidConnectionTokenError("invalid connection token")
        return int(value)

    @staticmethod
    def _invalid() -> InvalidConnectionTokenError:
        return InvalidConnectionTokenError("invalid connection token")
