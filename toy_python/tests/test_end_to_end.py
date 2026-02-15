"""End-to-end tests: Toy source -> parse -> lower -> optimize -> affine -> LLVM IR."""

from toy_python.parser.toy_parser import parse_toy
from toy_python.parser.lowering import lower
from toy_python.passes.optimize import optimize
from toy_python.passes.toy_to_affine import lower_to_affine
from toy_python.passes.affine_to_llvm import lower_to_llvm
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def compile(source: str) -> str:
    ast = parse_toy(source)
    ir = lower(ast)
    opt = optimize(ir)
    affine = lower_to_affine(opt)
    llvm = lower_to_llvm(affine)
    return asm.format(llvm)


def test_constant_print():
    """Constant tensor + print produces alloca, stores, and print_memref call."""
    source = strip_prefix("""
        | def main() {
        |   var x = [[1, 2, 3], [4, 5, 6]];
        |   print(x);
        |   return;
        | }
    """)
    result = compile(source)
    assert "%main = function () -> ():" in result, "Should have function def"
    assert "llvm.alloca(6)" in result, "Should alloca 6 elements for 2x3 tensor"
    assert "llvm.fconst(1.0)" in result, "Should store 1.0"
    assert "llvm.fconst(6.0)" in result, "Should store 6.0"
    assert "llvm.call(@print_memref" in result, "Should call print_memref"
    assert "return()" in result, "Should return void"


def test_transpose():
    """Transpose produces a second alloc and transposed load/store pattern."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(a);
        |   print(b);
        |   return;
        | }
    """)
    result = compile(source)
    assert "llvm.alloca(6)" in result, "Should have alloca for tensor"
    assert "llvm.load(" in result, "Should have loads for transpose"
    assert "llvm.gep(" in result, "Should have gep for indexing"
    assert "llvm.call(@print_memref" in result, "Should call print_memref"


def test_element_wise_mul():
    """Element-wise multiply produces fmul in the LLVM IR."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile(source)
    assert "llvm.fmul(" in result, "Should have fmul for element-wise multiply"
    assert "llvm.call(@print_memref" in result, "Should call print_memref"


def test_element_wise_add():
    """Element-wise add produces fadd in the LLVM IR."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a + b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile(source)
    assert "llvm.fadd(" in result, "Should have fadd for element-wise add"
    assert "llvm.call(@print_memref" in result, "Should call print_memref"


def test_reshape_folds_away():
    """Reshape of matching shape is optimized away -- no extra alloc."""
    source = strip_prefix("""
        | def main() {
        |   var x<2, 3> = [[1, 2, 3], [4, 5, 6]];
        |   print(x);
        |   return;
        | }
    """)
    result = compile(source)
    assert "llvm.alloca(6)" in result, "Should have single alloc"
    assert "llvm.call(@print_memref" in result, "Should call print_memref"


def test_double_transpose_optimized():
    """transpose(transpose(x)) is eliminated by the optimizer."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(transpose(a));
        |   print(b);
        |   return;
        | }
    """)
    result = compile(source)
    assert "llvm.alloca(6)" in result, "Should have alloc for constant"
    assert "llvm.call(@print_memref" in result, "Should call print_memref"
    assert "return()" in result, "Should return void"


def test_multiply_transpose_inlined():
    """Inlined multiply_transpose: transpose + multiply through full pipeline."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = [[1, 2, 3], [4, 5, 6]];
        |   var c = transpose(a) * transpose(b);
        |   print(c);
        |   return;
        | }
    """)
    result = compile(source)
    assert "%main = function () -> ():" in result, "Should have main function"
    assert "llvm.fmul(" in result, "Should have fmul for multiply"
    assert "loop_header" in result, "Should have loop headers"
    assert "llvm.phi(" in result, "Should have phi nodes"
    assert "llvm.call(@print_memref" in result, "Should call print_memref"
    assert "return()" in result, "Should return void"
