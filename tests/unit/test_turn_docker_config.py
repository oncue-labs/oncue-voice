from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]
TURN_CONFIG = REPOSITORY_ROOT / "docker" / "turn" / "coturn.conf"
TURN_README = REPOSITORY_ROOT / "docker" / "turn" / "README.md"
ENV_EXAMPLE = REPOSITORY_ROOT / ".env.template"


def test_coturn_reference_config_contains_safe_network_defaults() -> None:
    config = TURN_CONFIG.read_text()

    assert "listening-port=3478" in config
    assert "tls-listening-port=5349" in config
    assert "min-port=49152" in config
    assert "max-port=65535" in config
    assert "lt-cred-mech" in config
    assert "ONCUE_TURN_CREDENTIAL" not in config
    assert "temporary-credential" not in config


def test_coturn_readme_documents_shared_voice_settings() -> None:
    readme = TURN_README.read_text()
    env_example = ENV_EXAMPLE.read_text()

    for variable in (
        "ONCUE_VOICE_TURN_URLS",
        "ONCUE_VOICE_TURN_USERNAME",
        "ONCUE_VOICE_TURN_CREDENTIAL",
        "ONCUE_VOICE_TURN_USERNAME_FILE",
        "ONCUE_VOICE_TURN_CREDENTIAL_FILE",
        "ONCUE_TURN_REALM",
        "ONCUE_TURN_LISTENING_PORT",
        "ONCUE_TURN_TLS_LISTENING_PORT",
        "ONCUE_TURN_MIN_PORT",
        "ONCUE_TURN_MAX_PORT",
    ):
        assert variable in readme
        assert variable in env_example
