"""Phase 3 tests: Toy source -> IR -> text, compare against expected."""

import dgen
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.test.helpers import strip_prefix


def compile_toy(source: str) -> dgen.Value:
    ast = parse_toy(source)
    return lower(ast)


def test_simple_constant(ir_snapshot):
    source = strip_prefix("""
        | def main() {
        |   var x = [[1, 2, 3], [4, 5, 6]];
        |   print(x);
        |   return;
        | }
    """)
    assert compile_toy(source) == ir_snapshot


def test_explicit_shape_with_reshape(ir_snapshot):
    source = strip_prefix("""
        | def main() {
        |   var x = [[1, 2, 3], [4, 5, 6]];
        |   var y<2, 3> = x;
        |   print(y);
        |   return;
        | }
    """)
    assert compile_toy(source) == ir_snapshot


def test_binary_operations(ir_snapshot):
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    assert compile_toy(source) == ir_snapshot


def test_transpose_builtin(ir_snapshot):
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(a);
        |   print(b);
        |   return;
        | }
    """)
    assert compile_toy(source) == ir_snapshot


def test_generic_call(ir_snapshot):
    source = strip_prefix("""
        | def multiply_transpose(a, b) {
        |   return transpose(a) * transpose(b);
        | }
        |
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = [[1, 2, 3], [4, 5, 6]];
        |   var c = multiply_transpose(a, b);
        |   print(c);
        |   return;
        | }
    """)
    assert compile_toy(source) == ir_snapshot


def test_3d_constant(ir_snapshot):
    source = strip_prefix("""
        | def main() {
        |   var x = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   print(x);
        |   return;
        | }
    """)
    assert compile_toy(source) == ir_snapshot


def test_3d_binary_operations(ir_snapshot):
    source = strip_prefix("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a + b;
        |   print(c);
        |   return;
        | }
    """)
    assert compile_toy(source) == ir_snapshot


def test_full_tutorial_example(ir_snapshot):
    """The complete multiply_transpose example from the MLIR Toy tutorial."""
    source = strip_prefix("""
        | def multiply_transpose(a, b) {
        |   return transpose(a) * transpose(b);
        | }
        |
        | def main() {
        |   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
        |   var b<2, 3> = [1, 2, 3, 4, 5, 6];
        |   var c = multiply_transpose(a, b);
        |   var d = multiply_transpose(b, a);
        |   print(d);
        |   return;
        | }
    """)
    assert compile_toy(source) == ir_snapshot


def test_void_function_bare_return(ir_snapshot):
    """Function with only a bare return; — no ops, no side effects."""
    source = strip_prefix("""
        | def main() {
        |   return;
        | }
    """)
    assert compile_toy(source) == ir_snapshot


def test_tile_builtin(ir_snapshot):
    """tile() lowers to TileOp with count as parameter and input as operand."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = tile(a, 3);
        |   print(b);
        |   return;
        | }
    """)
    assert compile_toy(source) == ir_snapshot
