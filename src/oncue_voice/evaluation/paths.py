import re
from dataclasses import dataclass
from pathlib import Path


_RUN_ID_PATTERN = re.compile(r"^run-(?P<number>\d+)$")


@dataclass(frozen=True)
class EvaluationRunPaths:
    run_id: str
    run_directory: Path
    input_directory: Path
    artifact_directory: Path


def allocate_evaluation_run(
    artifact_root: Path,
    combination_key: str,
) -> EvaluationRunPaths:
    combination_directory = artifact_root / combination_key
    combination_directory.mkdir(parents=True, exist_ok=True)
    run_id = _next_run_id(combination_directory)
    run_directory = combination_directory / run_id
    input_directory = run_directory / "input"
    artifact_directory = run_directory / "artifacts"
    input_directory.mkdir(parents=True)
    artifact_directory.mkdir()
    return EvaluationRunPaths(
        run_id=run_id,
        run_directory=run_directory,
        input_directory=input_directory,
        artifact_directory=artifact_directory,
    )


def _next_run_id(combination_directory: Path) -> str:
    run_numbers = [
        match.group("number")
        for path in combination_directory.iterdir()
        if path.is_dir()
        and (match := _RUN_ID_PATTERN.fullmatch(path.name)) is not None
    ]
    next_number = max((int(number) for number in run_numbers), default=0) + 1
    return f"run-{next_number:03d}"
