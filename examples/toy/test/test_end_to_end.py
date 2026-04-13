"""End-to-end tests: Toy source through full pipeline.

IR snapshot tests verify the lowered LLVM-dialect IR.
JIT tests verify actual execution output.
"""

import llvmlite.binding as llvm_binding

import dgen
from dgen.llvm.codegen import emit_llvm_ir
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.ndbuffer_to_memory import NDBufferToMemory
from dgen.passes.record_to_memory import RecordToMemory
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.optimize import ToyOptimize
from toy.passes.shape_inference import ShapeInference
from toy.passes.toy_to_structured import ToyToStructured
from toy.test.helpers import run_toy as _toy, strip_prefix

compiler = Compiler(
    passes=[
        ToyOptimize(),
        ShapeInference(),
        ToyToStructured(),
        ControlFlowToGoto(),
        NDBufferToMemory(),
        RecordToMemory(),
        MemoryToLLVM(),
    ],
    exit=IdentityPass(),
)


def compile(source: str) -> dgen.Value:
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


# ---- JIT execution tests ----


class LowerLLVMToLLVMIR:
    def run(self, value: dgen.Value) -> str:
        llvm_ir, _ = emit_llvm_ir(value)
        return llvm_ir


def test_transpose_phi_emission():
    """Block args on labels produce phi instructions in LLVM IR."""
    source = """
    def main() {
      var a = [[1, 2, 3], [4, 5, 6]];
      var b = transpose(a);
      print(b);
      return;
    }
    """
    ast_node = parse_toy(source)
    ir = lower(ast_node)
    jit_compiler = Compiler(
        passes=[
            ToyOptimize(),
            ShapeInference(),
            ToyToStructured(),
            ControlFlowToGoto(),
            NDBufferToMemory(),
            RecordToMemory(),
            MemoryToLLVM(),
            AlgebraToLLVM(),
        ],
        exit=LowerLLVMToLLVMIR(),
    )
    llvm_ir = jit_compiler.compile(ir)
    llvm_binding.initialize_native_target()
    mod = llvm_binding.parse_assembly(llvm_ir)
    mod.verify()


def test_jit_constant_print():
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


def test_jit_transpose():
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


def test_jit_element_wise_mul():
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


def test_jit_element_wise_add():
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


def test_jit_3d_constant_print():
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


def test_jit_3d_element_wise_add():
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


def test_jit_3d_element_wise_mul():
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


def test_jit_double_transpose_optimized():
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
