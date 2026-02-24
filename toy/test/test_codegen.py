"""Tests for codegen: full pipeline with JIT execution."""

import tempfile

from click.testing import CliRunner

from toy.cli import main


def _toy(source):
    """Write .toy source to a temp file and run via CliRunner."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toy") as f:
        f.write(source)
        f.flush()
        r = CliRunner().invoke(main, [f.name])
        assert r.exit_code == 0, r.output
        return r.output.strip()


def test_constant_print():
    """Constant 2x3 tensor printed as flat values."""
    assert (
        _toy("""
        def main() {
          var x = [[1, 2, 3], [4, 5, 6]];
          print(x);
          return;
        }
    """)
        == "1, 2, 3, 4, 5, 6"
    )


def test_transpose():
    """Transpose 2x3 -> 3x2: row-major order changes."""
    assert (
        _toy("""
        def main() {
          var a = [[1, 2, 3], [4, 5, 6]];
          var b = transpose(a);
          print(b);
          return;
        }
    """)
        == "1, 4, 2, 5, 3, 6"
    )


def test_element_wise_mul():
    """Element-wise multiply of two 2x2 tensors."""
    assert (
        _toy("""
        def main() {
          var a = [[1, 2], [3, 4]];
          var b = [[5, 6], [7, 8]];
          var c = a * b;
          print(c);
          return;
        }
    """)
        == "5, 12, 21, 32"
    )


def test_element_wise_add():
    """Element-wise add of two 2x2 tensors."""
    assert (
        _toy("""
        def main() {
          var a = [[1, 2], [3, 4]];
          var b = [[5, 6], [7, 8]];
          var c = a + b;
          print(c);
          return;
        }
    """)
        == "6, 8, 10, 12"
    )


def test_3d_constant_print():
    """3D constant tensor printed as flat values."""
    assert (
        _toy("""
        def main() {
          var x = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
          print(x);
          return;
        }
    """)
        == "1, 2, 3, 4, 5, 6, 7, 8"
    )


def test_3d_element_wise_add():
    """Element-wise add of two 2x2x2 tensors."""
    assert (
        _toy("""
        def main() {
          var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
          var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
          var c = a + b;
          print(c);
          return;
        }
    """)
        == "3, 5, 7, 9, 11, 13, 15, 17"
    )


def test_3d_element_wise_mul():
    """Element-wise multiply of two 2x2x2 tensors."""
    assert (
        _toy("""
        def main() {
          var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
          var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
          var c = a * b;
          print(c);
          return;
        }
    """)
        == "2, 6, 12, 20, 30, 42, 56, 72"
    )


def test_double_transpose_optimized():
    """transpose(transpose(x)) optimized away — same output as original."""
    assert (
        _toy("""
        def main() {
          var a = [[1, 2, 3], [4, 5, 6]];
          var b = transpose(transpose(a));
          print(b);
          return;
        }
    """)
        == "1, 2, 3, 4, 5, 6"
    )
