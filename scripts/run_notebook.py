#!/usr/bin/env python3
"""Выполнить recommendations.ipynb и сохранить outputs (без Jupyter UI)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "patch_notebook_light.py")],
        cwd=ROOT,
        check=True,
    )
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(ROOT / "recommendations.ipynb"),
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    print("Notebook executed successfully.")


if __name__ == "__main__":
    main()
