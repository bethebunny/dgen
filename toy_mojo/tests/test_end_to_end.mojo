"""End-to-end tests: Toy source -> parse -> lower -> optimize -> affine -> LLVM IR."""

from testing import assert_true, TestSuite

from toy.parser.toy_parser import parse_toy
from toy.parser.lowering import lower
from toy.passes.optimize import optimize
from toy.passes.toy_to_affine import lower_to_affine
from toy.passes.affine_to_llvm import lower_to_llvm
from toy.dialects.llvm_printer import print_llvm_module


fn compile(source: String) raises -> String:
    var ast = parse_toy(source)
    var ir = lower(ast)
    var opt = optimize(ir)
    var affine = lower_to_affine(opt)
    var llvm = lower_to_llvm(affine)
    return print_llvm_module(llvm)


def test_constant_print():
    """Constant tensor + print produces alloca, stores, and print_memref call."""
    var source = String(
        'def main() {\n'
        '  var x = [[1, 2, 3], [4, 5, 6]];\n'
        '  print(x);\n'
        '  return;\n'
        '}\n'
    )
    var result = compile(source)
    assert_true("define void @main" in result, "Should have function def")
    assert_true("alloca f64, 6" in result, "Should alloca 6 elements for 2x3 tensor")
    assert_true("fconst 1.0" in result, "Should store 1.0")
    assert_true("fconst 6.0" in result, "Should store 6.0")
    assert_true("call @print_memref" in result, "Should call print_memref")
    assert_true("ret void" in result, "Should return void")


def test_transpose():
    """Transpose produces a second alloc and transposed load/store pattern."""
    var source = String(
        'def main() {\n'
        '  var a = [[1, 2, 3], [4, 5, 6]];\n'
        '  var b = transpose(a);\n'
        '  print(b);\n'
        '  return;\n'
        '}\n'
    )
    var result = compile(source)
    # Two allocs: one for constant (6 elems), one for transposed result (6 elems)
    assert_true("alloca f64, 6" in result, "Should have alloca for tensor")
    assert_true("load" in result, "Should have loads for transpose")
    assert_true("gep" in result, "Should have gep for indexing")
    assert_true("call @print_memref" in result, "Should call print_memref")


def test_element_wise_mul():
    """Element-wise multiply produces fmul in the LLVM IR."""
    var source = String(
        'def main() {\n'
        '  var a = [[1, 2], [3, 4]];\n'
        '  var b = [[5, 6], [7, 8]];\n'
        '  var c = a * b;\n'
        '  print(c);\n'
        '  return;\n'
        '}\n'
    )
    var result = compile(source)
    assert_true("fmul" in result, "Should have fmul for element-wise multiply")
    assert_true("call @print_memref" in result, "Should call print_memref")


def test_element_wise_add():
    """Element-wise add produces fadd in the LLVM IR."""
    var source = String(
        'def main() {\n'
        '  var a = [[1, 2], [3, 4]];\n'
        '  var b = [[5, 6], [7, 8]];\n'
        '  var c = a + b;\n'
        '  print(c);\n'
        '  return;\n'
        '}\n'
    )
    var result = compile(source)
    assert_true("fadd" in result, "Should have fadd for element-wise add")
    assert_true("call @print_memref" in result, "Should call print_memref")


def test_reshape_folds_away():
    """Reshape of matching shape is optimized away — no extra alloc."""
    var source = String(
        'def main() {\n'
        '  var x<2, 3> = [[1, 2, 3], [4, 5, 6]];\n'
        '  print(x);\n'
        '  return;\n'
        '}\n'
    )
    var result = compile(source)
    # Optimizer folds reshape of same-shape constant, so only one alloc
    assert_true("alloca f64, 6" in result, "Should have single alloc")
    assert_true("call @print_memref" in result, "Should call print_memref")


def test_double_transpose_optimized():
    """transpose(transpose(x)) is eliminated by the optimizer."""
    var source = String(
        'def main() {\n'
        '  var a = [[1, 2, 3], [4, 5, 6]];\n'
        '  var b = transpose(transpose(a));\n'
        '  print(b);\n'
        '  return;\n'
        '}\n'
    )
    var result = compile(source)
    # Double transpose is eliminated — no transposed load/store pattern needed.
    # Only one alloc for the constant.
    assert_true("alloca f64, 6" in result, "Should have alloc for constant")
    assert_true("call @print_memref" in result, "Should call print_memref")
    assert_true("ret void" in result, "Should return void")


def test_multiply_transpose_inlined():
    """Inlined multiply_transpose: transpose + multiply through full pipeline."""
    var source = String(
        'def main() {\n'
        '  var a = [[1, 2, 3], [4, 5, 6]];\n'
        '  var b = [[1, 2, 3], [4, 5, 6]];\n'
        '  var c = transpose(a) * transpose(b);\n'
        '  print(c);\n'
        '  return;\n'
        '}\n'
    )
    var result = compile(source)
    assert_true("define void @main" in result, "Should have main function")
    # Should have fmul from the Mul op
    assert_true("fmul" in result, "Should have fmul for multiply")
    # Should have loop structure from affine lowering
    assert_true("loop_header" in result, "Should have loop headers")
    assert_true("phi" in result, "Should have phi nodes")
    assert_true("call @print_memref" in result, "Should call print_memref")
    assert_true("ret void" in result, "Should return void")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
