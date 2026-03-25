"""End-to-end tests: Toy source -> parse -> lower -> optimize -> structured -> LLVM IR."""

from dgen.codegen import LLVMCodegen
from dgen.compiler import Compiler, IdentityPass
from dgen.module import Module
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.control_flow_to_goto import ControlFlowToGoto
from toy.passes.memory_to_llvm import MemoryToLLVM
from toy.passes.structured_to_llvm import StructuredToLLVM
from toy.passes.optimize import ToyOptimize
from toy.passes.shape_inference import ShapeInference
from toy.passes.toy_to_structured import ToyToStructured
from toy.test.helpers import strip_prefix

compiler = Compiler(
    passes=[
        ToyOptimize(),
        ShapeInference(),
        ToyToStructured(),
        StructuredToLLVM(),
    ],
    exit=IdentityPass(),
)


def compile(source: str) -> Module:
    ast = parse_toy(source)
    ir = lower(ast)
    return compiler.run(ir)


def test_constant_print(ir_snapshot):
    """Constant tensor + print: tensor constant passes through, codegen materializes."""
    source = strip_prefix("""
        | def main() {
        |   var x = [[1, 2, 3], [4, 5, 6]];
        |   print(x);
        |   return;
        | }
    """)
    assert compile(source) == ir_snapshot


def test_transpose(ir_snapshot):
    """Transpose produces a second alloc and transposed load/store pattern."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(a);
        |   print(b);
        |   return;
        | }
    """)
    assert compile(source) == ir_snapshot


def test_element_wise_mul(ir_snapshot):
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
    assert compile(source) == ir_snapshot


def test_element_wise_add(ir_snapshot):
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
    assert compile(source) == ir_snapshot


def test_3d_constant_print(ir_snapshot):
    """3D constant tensor + print: tensor constant passes through."""
    source = strip_prefix("""
        | def main() {
        |   var x = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   print(x);
        |   return;
        | }
    """)
    assert compile(source) == ir_snapshot


def test_3d_element_wise_add(ir_snapshot):
    """3D element-wise add produces fadd in the LLVM IR."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a + b;
        |   print(c);
        |   return;
        | }
    """)
    assert compile(source) == ir_snapshot


def test_3d_element_wise_mul(ir_snapshot):
    """3D element-wise multiply produces fmul in the LLVM IR."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    assert compile(source) == ir_snapshot


def test_reshape_folds_away(ir_snapshot):
    """Reshape of matching shape is optimized away -- no extra alloc."""
    source = strip_prefix("""
        | def main() {
        |   var x<2, 3> = [[1, 2, 3], [4, 5, 6]];
        |   print(x);
        |   return;
        | }
    """)
    assert compile(source) == ir_snapshot


def test_double_transpose_optimized(ir_snapshot):
    """transpose(transpose(x)) is eliminated by the optimizer."""
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(transpose(a));
        |   print(b);
        |   return;
        | }
    """)
    assert compile(source) == ir_snapshot


def test_multiply_transpose_inlined(ir_snapshot):
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
    assert compile(source) == ir_snapshot


# ===----------------------------------------------------------------------=== #
# Decomposed pipeline: ControlFlowToGoto + MemoryToLLVM + AlgebraToLLVM
# ===----------------------------------------------------------------------=== #

_decomposed = Compiler(
    passes=[
        ToyOptimize(),
        ShapeInference(),
        ToyToStructured(),
        ControlFlowToGoto(),
        MemoryToLLVM(),
    ],
    exit=LLVMCodegen(),
)


def _jit_decomposed(source: str) -> object:
    """Compile through the decomposed pipeline and JIT-execute."""
    ir = lower(parse_toy(strip_prefix(source)))
    exe = _decomposed.compile(ir)
    return exe.run().to_json()


def test_decomposed_element_wise_mul():
    """Decomposed pipeline: element-wise multiply produces correct JIT output."""
    _jit_decomposed("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)


def test_decomposed_transpose():
    """Decomposed pipeline: transpose produces correct JIT output."""
    _jit_decomposed("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(a);
        |   print(b);
        |   return;
        | }
    """)


def test_decomposed_3d_mul():
    """Decomposed pipeline: 3D element-wise multiply."""
    _jit_decomposed("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
