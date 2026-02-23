"""CLI tests: exercise the full pipeline (source/IR → JIT output).

Tests call cli.run (toy source) and cli.run_ir (IR text) — the same
code path used by the CLI entry point.
"""

from pathlib import Path

from click.testing import CliRunner

from toy.cli import main, run, run_ir
from toy.test.helpers import strip_prefix

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


# ===----------------------------------------------------------------------=== #
# cli.run — toy source through the full pipeline
# ===----------------------------------------------------------------------=== #


def test_run_constant():
    output = run(TESTDATA.joinpath("constant.toy").read_text(), capture_output=True)
    assert output is not None
    assert output.strip() == "1, 2, 3, 4, 5, 6"


def test_run_transpose():
    output = run(TESTDATA.joinpath("transpose.toy").read_text(), capture_output=True)
    assert output is not None
    assert output.strip() == "1, 4, 2, 5, 3, 6"


def test_run_multiply():
    output = run(TESTDATA.joinpath("multiply.toy").read_text(), capture_output=True)
    assert output is not None
    assert output.strip() == "5, 12, 21, 32"


# ===----------------------------------------------------------------------=== #
# cli.run_ir — IR text through the staging pipeline
# ===----------------------------------------------------------------------=== #


def test_ir_tile_constant():
    """tile with a literal count — no staging needed."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([3], f64) = [1.0, 2.0, 3.0]
        |     %1 : index = 2
        |     %2 : toy.Tensor([2, 3], f64) = toy.tile(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return()
    """)
    output = run_ir(ir, capture_output=True)
    assert output is not None
    assert output.strip() == "1, 2, 3, 1, 2, 3"


def test_ir_tile_computed_count():
    """tile count = add_index(1, 1) — stage-0 JIT."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([3], f64) = [7.0, 8.0, 9.0]
        |     %1 : index = 1
        |     %2 : index = 1
        |     %3 : index = add_index(%1, %2)
        |     %4 : toy.InferredShapeTensor(f64) = toy.tile(%0, %3)
        |     %5 : () = toy.print(%4)
        |     %_ : () = return()
    """)
    output = run_ir(ir, capture_output=True)
    assert output is not None
    assert output.strip() == "7, 8, 9, 7, 8, 9"


def test_ir_tile_nonzero_count_constant():
    """tile count = nonzero_count on constant tensor — stage-0 JIT."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor([4], f64) = [1.0, 0.0, 3.0, 0.0]
        |     %1 : index = toy.nonzero_count(%0)
        |     %2 : toy.Tensor([3], f64) = [7.0, 8.0, 9.0]
        |     %3 : toy.InferredShapeTensor(f64) = toy.tile(%2, %1)
        |     %4 : () = toy.print(%3)
        |     %_ : () = return()
    """)
    output = run_ir(ir, capture_output=True)
    assert output is not None
    assert output.strip() == "7, 8, 9, 7, 8, 9"


def test_ir_tile_nonzero_count_param():
    """tile count = nonzero_count on parameter — stage-1 JIT."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%x: toy.Tensor([4], f64)) -> ():
        |     %1 : index = toy.nonzero_count(%x)
        |     %2 : toy.Tensor([3], f64) = [7.0, 8.0, 9.0]
        |     %3 : toy.InferredShapeTensor(f64) = toy.tile(%2, %1)
        |     %4 : () = toy.print(%3)
        |     %_ : () = return()
    """)
    output = run_ir(ir, args=[[1.0, 0.0, 3.0, 0.0]], capture_output=True)
    assert output is not None
    assert output.strip() == "7, 8, 9, 7, 8, 9"


def test_ir_param_in_stage2():
    """Stage-2 uses the function parameter (pointer-crossing)."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%x: toy.Tensor([4], f64)) -> ():
        |     %1 : index = toy.nonzero_count(%x)
        |     %2 : toy.InferredShapeTensor(f64) = toy.tile(%x, %1)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return()
    """)
    output = run_ir(ir, args=[[1.0, 0.0, 3.0, 0.0]], capture_output=True)
    assert output is not None
    assert output.strip() == "1, 0, 3, 0, 1, 0, 3, 0"


def test_ir_two_staged_tiles():
    """Two tiles with independent runtime counts (multiple comptime)."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%x: toy.Tensor([4], f64), %y: toy.Tensor([3], f64)) -> ():
        |     %c1 : index = toy.nonzero_count(%x)
        |     %d1 : toy.Tensor([2], f64) = [1.0, 2.0]
        |     %t1 : toy.InferredShapeTensor(f64) = toy.tile(%d1, %c1)
        |     %p1 : () = toy.print(%t1)
        |     %c2 : index = toy.nonzero_count(%y)
        |     %d2 : toy.Tensor([2], f64) = [3.0, 4.0]
        |     %t2 : toy.InferredShapeTensor(f64) = toy.tile(%d2, %c2)
        |     %p2 : () = toy.print(%t2)
        |     %_ : () = return()
    """)
    output = run_ir(
        ir, args=[[1.0, 0.0, 3.0, 0.0], [1.0, 2.0, 0.0]], capture_output=True
    )
    assert output is not None
    lines = output.strip().split("\n")
    assert lines[0] == "1, 2, 1, 2"
    assert lines[1] == "3, 4, 3, 4"


def test_ir_chained_staged_tiles():
    """Second tile depends on first tile's resolved shape (arbitrary stages)."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function (%x: toy.Tensor([4], f64)) -> ():
        |     %c1 : index = toy.nonzero_count(%x)
        |     %base : toy.Tensor([2], f64) = [10.0, 20.0]
        |     %t1 : toy.InferredShapeTensor(f64) = toy.tile(%base, %c1)
        |     %len : index = toy.dim_size(%t1, 0)
        |     %d2 : toy.Tensor([1], f64) = [5.0]
        |     %t2 : toy.InferredShapeTensor(f64) = toy.tile(%d2, %len)
        |     %p : () = toy.print(%t2)
        |     %_ : () = return()
    """)
    output = run_ir(ir, args=[[1.0, 0.0, 3.0, 0.0]], capture_output=True)
    assert output is not None
    assert output.strip() == "5, 5"
