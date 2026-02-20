"""CLI tests: subprocess invocation on .toy test data files."""

import subprocess
import sys
from pathlib import Path

TESTDATA = Path(__file__).parent / "testdata"


def _run_cli(toy_file: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "toy.cli", str(toy_file)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_constant():
    r = _run_cli(TESTDATA / "constant.toy")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "1, 2, 3, 4, 5, 6"


def test_transpose():
    r = _run_cli(TESTDATA / "transpose.toy")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "1, 4, 2, 5, 3, 6"


def test_multiply():
    r = _run_cli(TESTDATA / "multiply.toy")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "5, 12, 21, 32"


def test_add():
    r = _run_cli(TESTDATA / "add.toy")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "6, 8, 10, 12"


def test_multiply_transpose():
    r = _run_cli(TESTDATA / "multiply_transpose.toy")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "1, 16, 4, 25, 9, 36"
