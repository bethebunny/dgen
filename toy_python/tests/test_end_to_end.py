"""End-to-end tests: Toy source -> parse -> lower -> optimize -> affine -> LLVM IR."""

from toy_python.parser.toy_parser import parse_toy
from toy_python.parser.lowering import lower
from toy_python.passes.optimize import optimize
from toy_python.passes.toy_to_affine import lower_to_affine
from toy_python.passes.affine_to_llvm import lower_to_llvm
from toy_python import asm


def compile(source: str) -> str:
    ast = parse_toy(source)
    ir = lower(ast)
    opt = optimize(ir)
    affine = lower_to_affine(opt)
    llvm = lower_to_llvm(affine)
    return asm.format(llvm)


def test_constant_print():
    """Constant tensor + print produces alloca, stores, and print_memref call."""
    source = (
        "def main() {\n"
        "  var x = [[1, 2, 3], [4, 5, 6]];\n"
        "  print(x);\n"
        "  return;\n"
        "}\n"
    )
    result = compile(source)
    assert "%main = function () -> ():" in result, "Should have function def"
    assert "alloca(6)" in result, "Should alloca 6 elements for 2x3 tensor"
    assert "fconst(1.0)" in result, "Should store 1.0"
    assert "fconst(6.0)" in result, "Should store 6.0"
    assert "call(@print_memref" in result, "Should call print_memref"
    assert "return()" in result, "Should return void"


def test_transpose():
    """Transpose produces a second alloc and transposed load/store pattern."""
    source = (
        "def main() {\n"
        "  var a = [[1, 2, 3], [4, 5, 6]];\n"
        "  var b = transpose(a);\n"
        "  print(b);\n"
        "  return;\n"
        "}\n"
    )
    result = compile(source)
    assert "alloca(6)" in result, "Should have alloca for tensor"
    assert "load(" in result, "Should have loads for transpose"
    assert "gep(" in result, "Should have gep for indexing"
    assert "call(@print_memref" in result, "Should call print_memref"


def test_element_wise_mul():
    """Element-wise multiply produces fmul in the LLVM IR."""
    source = (
        "def main() {\n"
        "  var a = [[1, 2], [3, 4]];\n"
        "  var b = [[5, 6], [7, 8]];\n"
        "  var c = a * b;\n"
        "  print(c);\n"
        "  return;\n"
        "}\n"
    )
    result = compile(source)
    assert "fmul(" in result, "Should have fmul for element-wise multiply"
    assert "call(@print_memref" in result, "Should call print_memref"


def test_element_wise_add():
    """Element-wise add produces fadd in the LLVM IR."""
    source = (
        "def main() {\n"
        "  var a = [[1, 2], [3, 4]];\n"
        "  var b = [[5, 6], [7, 8]];\n"
        "  var c = a + b;\n"
        "  print(c);\n"
        "  return;\n"
        "}\n"
    )
    result = compile(source)
    assert "fadd(" in result, "Should have fadd for element-wise add"
    assert "call(@print_memref" in result, "Should call print_memref"


def test_reshape_folds_away():
    """Reshape of matching shape is optimized away -- no extra alloc."""
    source = (
        "def main() {\n"
        "  var x<2, 3> = [[1, 2, 3], [4, 5, 6]];\n"
        "  print(x);\n"
        "  return;\n"
        "}\n"
    )
    result = compile(source)
    assert "alloca(6)" in result, "Should have single alloc"
    assert "call(@print_memref" in result, "Should call print_memref"


def test_double_transpose_optimized():
    """transpose(transpose(x)) is eliminated by the optimizer."""
    source = (
        "def main() {\n"
        "  var a = [[1, 2, 3], [4, 5, 6]];\n"
        "  var b = transpose(transpose(a));\n"
        "  print(b);\n"
        "  return;\n"
        "}\n"
    )
    result = compile(source)
    assert "alloca(6)" in result, "Should have alloc for constant"
    assert "call(@print_memref" in result, "Should call print_memref"
    assert "return()" in result, "Should return void"


def test_multiply_transpose_inlined():
    """Inlined multiply_transpose: transpose + multiply through full pipeline."""
    source = (
        "def main() {\n"
        "  var a = [[1, 2, 3], [4, 5, 6]];\n"
        "  var b = [[1, 2, 3], [4, 5, 6]];\n"
        "  var c = transpose(a) * transpose(b);\n"
        "  print(c);\n"
        "  return;\n"
        "}\n"
    )
    result = compile(source)
    assert "%main = function () -> ():" in result, "Should have main function"
    assert "fmul(" in result, "Should have fmul for multiply"
    assert "loop_header" in result, "Should have loop headers"
    assert "phi(" in result, "Should have phi nodes"
    assert "call(@print_memref" in result, "Should call print_memref"
    assert "return()" in result, "Should return void"
