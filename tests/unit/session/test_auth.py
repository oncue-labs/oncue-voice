from datetime import datetime, timezone

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from oncue_voice.session.auth import (
    ConnectionTokenVerifier,
    ConnectionTokenVerifierOptions,
    InvalidConnectionTokenError,
)
from oncue_voice.session.store import InMemoryJtiStore


def create_key_pair() -> tuple[str, str]:
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


def create_token(private_key: str, **overrides: object) -> str:
    now = int(datetime.now(timezone.utc).timestamp())
    claims: dict[str, object] = {
        "callSessionId": 1,
        "userId": 7,
        "scope": "voice:connect",
        "jti": "token-1",
        "iat": now,
        "exp": now + 60,
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256")


def create_verifier(public_key: str) -> ConnectionTokenVerifier:
    return ConnectionTokenVerifier(public_key, InMemoryJtiStore())


def test_verifier_accepts_signed_token_and_consumes_jti_once() -> None:
    private_key, public_key = create_key_pair()
    token = create_token(private_key)
    verifier = create_verifier(public_key)

    claims = verifier.verify(token, 1, 7)

    assert claims.call_session_id == 1
    assert claims.user_id == 7
    assert claims.scope == ("voice:connect",)
    assert claims.jti == "token-1"


def test_verifier_rejects_expired_token() -> None:
    private_key, public_key = create_key_pair()
    token = create_token(private_key, exp=1)

    with pytest.raises(InvalidConnectionTokenError, match="invalid connection token"):
        create_verifier(public_key).verify(token, 1, 7)


def test_verifier_rejects_token_for_another_call_session() -> None:
    private_key, public_key = create_key_pair()
    token = create_token(private_key)

    with pytest.raises(InvalidConnectionTokenError, match="invalid connection token"):
        create_verifier(public_key).verify(token, 2, 7)


def test_verifier_rejects_token_for_another_user() -> None:
    private_key, public_key = create_key_pair()
    token = create_token(private_key)

    with pytest.raises(InvalidConnectionTokenError, match="invalid connection token"):
        create_verifier(public_key).verify(token, 1, 8)


def test_verifier_rejects_missing_required_scope() -> None:
    private_key, public_key = create_key_pair()
    token = create_token(private_key, scope="reservation:read")

    with pytest.raises(InvalidConnectionTokenError, match="invalid connection token"):
        create_verifier(public_key).verify(token, 1, 7)


def test_verifier_rejects_replayed_jti() -> None:
    private_key, public_key = create_key_pair()
    token = create_token(private_key)
    verifier = create_verifier(public_key)

    verifier.verify(token, 1, 7)

    with pytest.raises(InvalidConnectionTokenError, match="invalid connection token"):
        verifier.verify(token, 1, 7)


def test_verifier_supports_injected_clock_for_expiration() -> None:
    private_key, public_key = create_key_pair()
    token = create_token(
        private_key,
        iat=1_000,
        exp=1_060,
    )
    verifier = ConnectionTokenVerifier(
        public_key,
        InMemoryJtiStore(clock=lambda: 1_010.0),
        ConnectionTokenVerifierOptions(clock=lambda: 1_010.0),
    )

    claims = verifier.verify(token, 1, 7)

    assert claims.exp == 1_060


def test_verifier_rejects_string_identifiers() -> None:
    private_key, public_key = create_key_pair()
    token = create_token(private_key, callSessionId="1", userId="7")

    with pytest.raises(InvalidConnectionTokenError, match="invalid connection token"):
        create_verifier(public_key).verify(token, 1, 7)
