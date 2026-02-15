"""Tests for codegen: full pipeline with JIT execution."""

from toy_python.parser.toy_parser import parse_toy
from toy_python.parser.lowering import lower
from toy_python.passes.optimize import optimize
from toy_python.passes.toy_to_affine import lower_to_affine
from toy_python.passes.affine_to_llvm import lower_to_llvm
from toy_python.codegen import compile_and_run
from toy_python.tests.helpers import strip_prefix


def run_toy(source: str) -> str:
    """Full pipeline: source -> parse -> lower -> optimize -> affine -> llvm -> JIT."""
    ast = parse_toy(source)
    ir = lower(ast)
    opt = optimize(ir)
    affine = lower_to_affine(opt)
    ll = lower_to_llvm(affine)
    output = compile_and_run(ll, capture_output=True)
    assert output is not None
    return output


def test_constant_print():
    """Constant 2x3 tensor printed as flat values."""
    source = strip_prefix("""
        | def main() {
        |   var x = [[1, 2, 3], [4, 5, 6]];
        |   print(x);
        |   return;
        | }
    """)
    output = run_toy(source)
    assert output.strip() == "1, 2, 3, 4, 5, 6"


def test_transpose():
    """Transpose 2x3 -> 3x2: row-major order changes."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(a);
        |   print(b);
        |   return;
        | }
    """)
    output = run_toy(source)
    assert output.strip() == "1, 4, 2, 5, 3, 6"


def test_element_wise_mul():
    """Element-wise multiply of two 2x2 tensors."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    output = run_toy(source)
    assert output.strip() == "5, 12, 21, 32"


def test_element_wise_add():
    """Element-wise add of two 2x2 tensors."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a + b;
        |   print(c);
        |   return;
        | }
    """)
    output = run_toy(source)
    assert output.strip() == "6, 8, 10, 12"


def test_double_transpose_optimized():
    """transpose(transpose(x)) optimized away — same output as original."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(transpose(a));
        |   print(b);
        |   return;
        | }
    """)
    output = run_toy(source)
    assert output.strip() == "1, 2, 3, 4, 5, 6"
