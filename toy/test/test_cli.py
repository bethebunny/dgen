"""CLI tests: in-process invocation on .toy test data files."""

from pathlib import Path

from click.testing import CliRunner

from toy.cli import main

TESTDATA = Path(__file__).parent / "testdata"


def _run_cli(toy_file: Path) -> CliRunner.invoke:
    return CliRunner().invoke(main, [str(toy_file)])


def test_constant():
    r = _run_cli(TESTDATA / "constant.toy")
    assert r.exit_code == 0, r.output
    assert r.output.strip() == "1, 2, 3, 4, 5, 6"


def test_transpose():
    r = _run_cli(TESTDATA / "transpose.toy")
    assert r.exit_code == 0, r.output
    assert r.output.strip() == "1, 4, 2, 5, 3, 6"


def test_multiply():
    r = _run_cli(TESTDATA / "multiply.toy")
    assert r.exit_code == 0, r.output
    assert r.output.strip() == "5, 12, 21, 32"


def test_add():
    r = _run_cli(TESTDATA / "add.toy")
    assert r.exit_code == 0, r.output
    assert r.output.strip() == "6, 8, 10, 12"


def test_multiply_transpose():
    r = _run_cli(TESTDATA / "multiply_transpose.toy")
    assert r.exit_code == 0, r.output
    assert r.output.strip() == "1, 16, 4, 25, 9, 36"
