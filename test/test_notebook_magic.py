"""Tests for the %%dgen-dialect notebook cell magic.

Executes a real Jupyter notebook (test/testdata/dialect_magic.ipynb) via
papermill so that the IPython magic machinery is exercised end-to-end.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_NOTEBOOK = _REPO_ROOT / "test" / "testdata" / "dialect_magic.ipynb"


def test_dgen_dialect_magic_in_notebook(tmp_path: Path) -> None:
    """%%dgen-dialect cell magic compiles a dialect and injects it into the kernel."""
    output = tmp_path / "dialect_magic_out.ipynb"
    # Ensure the repo root is on PYTHONPATH so the kernel subprocess can
    # import dgen even when the package is not pip-installed.
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_REPO_ROOT}:{pythonpath}" if pythonpath else str(_REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "papermill",
            str(_NOTEBOOK),
            str(output),
            "--no-progress-bar",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Notebook execution failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
