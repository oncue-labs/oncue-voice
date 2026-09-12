import json
from pathlib import Path


NOTEBOOKS = (
    Path("notebooks/voice_persona_scenario_evaluation.ipynb"),
    Path("notebooks/realtime_persona_scenario_evaluation.ipynb"),
)


def test_evaluation_notebooks_load_project_environment_file() -> None:
    for notebook_path in NOTEBOOKS:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        assert "from dotenv import load_dotenv" in source
        assert "load_dotenv()" in source


def test_environment_example_contains_no_secret_value() -> None:
    example = Path(".env.example").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY=" in example
    assert "OPENAI_API_KEY=sk-" not in example
