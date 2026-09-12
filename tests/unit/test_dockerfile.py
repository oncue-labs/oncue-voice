from pathlib import Path


def test_dockerfile_installs_dependencies_before_copying_runtime_source() -> None:
    dockerfile = (Path(__file__).parents[2] / "Dockerfile").read_text()

    assert "poetry install --only main --no-root" in dockerfile
    assert "PYTHONPATH=/app/src" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert '"oncue_voice.main:create_app"' in dockerfile
