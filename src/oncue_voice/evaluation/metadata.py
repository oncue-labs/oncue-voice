import subprocess
from pathlib import Path


def get_oncue_voice_commit() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()
