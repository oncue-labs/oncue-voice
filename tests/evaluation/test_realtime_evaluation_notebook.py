import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


NOTEBOOK_PATH = Path("notebooks/realtime_persona_scenario_evaluation.ipynb")


def test_realtime_evaluation_notebook_runs_four_combinations_with_one_run_each(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ONCUE_EVALUATION_PROVIDER", "fake")
    monkeypatch.setenv("ONCUE_EVALUATION_ARTIFACT_ROOT", str(tmp_path))

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent.parent)}},
    )

    client.execute()

    run_files = sorted(tmp_path.glob("*/*/artifacts/run.json"))
    assert len(run_files) == 4
    for run_file in run_files:
        run_data = json.loads(run_file.read_text(encoding="utf-8"))
        assert run_data["evaluationPath"] == "realtime"
        assert run_data["succeeded"] is True
        assert "variantId" not in run_data
        provider_config = run_file.parent / "provider-config.json"
        assert "apiKey" not in provider_config.read_text(encoding="utf-8")
        assert (run_file.parent.parent / "input" / "input.pcm").exists()
