"""Staging tests: dependent types that require compile-time evaluation.

These tests verify that the staging evaluator resolves Constant fields
so that shape inference and lowering can proceed. All tests exercise
the full pipeline (source → CLI → staging → shape inference → lowering → JIT).
"""

from toy.test.helpers import run_toy as _toy


# ===----------------------------------------------------------------------=== #
# Stage-0: comptime values from constants
# ===----------------------------------------------------------------------=== #


def test_tile_constant():
    """tile([1, 2, 3], 2) prints two copies."""
    assert (
        _toy("""
        def main() {
            var data = [1, 2, 3];
            var t = tile(data, 2);
            print(t);
            return;
        }
    """)
        == "1, 2, 3, 1, 2, 3"
    )


def test_tile_constant_3copies():
    """tile([4, 5], 3) prints three copies."""
    assert (
        _toy("""
        def main() {
            var data = [4, 5];
            var t = tile(data, 3);
            print(t);
            return;
        }
    """)
        == "4, 5, 4, 5, 4, 5"
    )


def test_tile_add_index():
    """tile count = add_index(2, 2) → 4 copies."""
    assert (
        _toy("""
        def main() {
            var data = [1, 2, 3];
            var t = tile(data, add_index(2, 2));
            print(t);
            return;
        }
    """)
        == "1, 2, 3, " * 3 + "1, 2, 3"
    )


def test_tile_chained_add():
    """tile count = add_index(add_index(1, 1), add_index(1, 1)) → 4 copies."""
    assert (
        _toy("""
        def main() {
            var data = [1, 2, 3];
            var count = add_index(add_index(1, 1), add_index(1, 1));
            var t = tile(data, count);
            print(t);
            return;
        }
    """)
        == "1, 2, 3, " * 3 + "1, 2, 3"
    )


def test_tile_nonzero_count():
    """tile count = nonzero_count on constant tensor → 2 copies."""
    assert (
        _toy("""
        def main() {
            var mask = [1, 0, 3, 0];
            var n = nonzero_count(mask);
            var base = [7, 8, 9];
            var t = tile(base, n);
            print(t);
            return;
        }
    """)
        == "7, 8, 9, 7, 8, 9"
    )


def test_tile_nonzero_plus_add():
    """tile count = add_index(nonzero_count([1,0,3,0]), 1) → 3 copies."""
    assert (
        _toy("""
        def main() {
            var mask = [1, 0, 3, 0];
            var n = nonzero_count(mask);
            var count = add_index(n, 1);
            var base = [5, 6];
            var t = tile(base, count);
            print(t);
            return;
        }
    """)
        == "5, 6, 5, 6, 5, 6"
    )


def test_tile_shape_propagates_to_mul():
    """tile shape propagates through downstream mul.

    tile(Tensor([3]), 4) * tile(Tensor([3]), 4) → Tensor([4, 3])
    """
    assert (
        _toy("""
        def main() {
            var data = [1, 2, 3];
            var count = add_index(2, 2);
            var a = tile(data, count);
            var b = tile(data, count);
            print(a * b);
            return;
        }
    """)
        == "1, 4, 9, " * 3 + "1, 4, 9"
    )


def test_concat_constant():
    """concat two tensors along axis 0."""
    assert (
        _toy("""
        def main() {
            var a = [[1, 2, 3], [4, 5, 6]];
            var b = [[7, 8, 9]];
            var c = concat(a, b, 0);
            print(c);
            return;
        }
    """)
        == "1, 2, 3, 4, 5, 6, 7, 8, 9"
    )


def test_concat_after_computed_tile():
    """concat with a tile whose count is computed via staging.

    tile(Tensor([3]), add_index(2, 1)) → Tensor([3, 3])
    concat(Tensor([2, 3]), Tensor([3, 3]), axis=0) → Tensor([5, 3])
    """
    assert (
        _toy("""
        def main() {
            var a = [[1, 2, 3], [4, 5, 6]];
            var b = [7, 8, 9];
            var n = add_index(2, 1);
            var t = tile(b, n);
            var c = concat(a, t, 0);
            print(c);
            return;
        }
    """)
        == "1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9"
    )


# ===----------------------------------------------------------------------=== #
# Stage-1: runtime-dependent comptime values
# ===----------------------------------------------------------------------=== #


def test_stage1_nonzero_count():
    """nonzero_count on a function parameter — stage-1 JIT resolves tile shape."""
    assert (
        _toy(
            """
        def main(x) {
            var n = nonzero_count(x);
            var base = [7, 8, 9];
            var t = tile(base, n);
            print(t);
            return;
        }
    """,
            args=[[1.0, 0.0, 3.0, 0.0]],
        )
        == "7, 8, 9, 7, 8, 9"
    )


def test_stage1_nonzero_count_different_input():
    """Different tensor, different nonzero count — stage-1 adapts."""
    assert (
        _toy(
            """
        def main(x) {
            var n = nonzero_count(x);
            var base = [7, 8, 9];
            var t = tile(base, n);
            print(t);
            return;
        }
    """,
            args=[[1.0, 2.0, 3.0, 4.0]],
        )
        == "7, 8, 9, " * 3 + "7, 8, 9"
    )


def test_stage1_param_in_stage2():
    """Stage-2 uses the original function parameter (pointer-crossing)."""
    assert (
        _toy(
            """
        def main(x) {
            var n = nonzero_count(x);
            var t = tile(x, n);
            print(t);
            return;
        }
    """,
            args=[[1.0, 0.0, 3.0, 0.0]],
        )
        == "1, 0, 3, 0, 1, 0, 3, 0"
    )


def test_stage1_two_tiles():
    """Two TileOps with independent runtime-dependent counts."""
    output = _toy(
        """
        def main(x, y) {
            var c1 = nonzero_count(x);
            var d1 = [1, 2];
            var t1 = tile(d1, c1);
            print(t1);
            var c2 = nonzero_count(y);
            var d2 = [3, 4];
            var t2 = tile(d2, c2);
            print(t2);
            return;
        }
    """,
        args=[[1.0, 0.0, 3.0, 0.0], [1.0, 2.0, 0.0]],
    )
    lines = output.split("\n")
    assert lines[0] == "1, 2, 1, 2"
    assert lines[1] == "3, 4, 3, 4"


def test_stage1_chained_nonzero_tile():
    """Second tile count depends on shape resolved by the first tile."""
    assert (
        _toy(
            """
        def main(x) {
            var c1 = nonzero_count(x);
            var base = [10, 20];
            var t1 = tile(base, c1);
            var len = dim_size(t1, 0);
            var d2 = [5];
            var t2 = tile(d2, len);
            print(t2);
            return;
        }
    """,
            args=[[1.0, 0.0, 3.0, 0.0]],
        )
        == "5, 5"
    )


# ===----------------------------------------------------------------------=== #
# String args (CLI path): args passed as ASM literal strings
# ===----------------------------------------------------------------------=== #


def test_stage1_string_arg():
    """String arg parsed via Memory.from_asm in _prepare_ctypes_args."""
    assert (
        _toy(
            """
        def main(x) {
            var n = nonzero_count(x);
            var base = [7, 8, 9];
            var t = tile(base, n);
            print(t);
            return;
        }
    """,
            args=["[1.0, 0.0, 3.0, 0.0]"],
        )
        == "7, 8, 9, 7, 8, 9"
    )


def test_stage1_two_string_args():
    """Two string args with independent runtime-dependent counts."""
    output = _toy(
        """
        def main(x, y) {
            var c1 = nonzero_count(x);
            var d1 = [1, 2];
            var t1 = tile(d1, c1);
            print(t1);
            var c2 = nonzero_count(y);
            var d2 = [3, 4];
            var t2 = tile(d2, c2);
            print(t2);
            return;
        }
    """,
        args=["[1.0, 0.0, 3.0, 0.0]", "[1.0, 2.0, 0.0]"],
    )
    lines = output.split("\n")
    assert lines[0] == "1, 2, 1, 2"
    assert lines[1] == "3, 4, 3, 4"
