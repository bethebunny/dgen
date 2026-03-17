"""Tests for codegen: full pipeline with JIT execution."""

from dgen.codegen import emit_llvm_ir
from dgen.passes.builtin_to_llvm import lower_builtin_to_llvm
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.affine_to_llvm import lower_to_llvm
from toy.passes.optimize import optimize
from toy.passes.shape_inference import infer_shapes
from toy.passes.toy_to_affine import lower_to_affine
from toy.test.helpers import run_toy as _toy


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
    opt = optimize(ir)
    typed = infer_shapes(opt)
    affine_mod = lower_to_affine(typed)
    llvm_mod = lower_to_llvm(affine_mod)
    llvm_mod = lower_builtin_to_llvm(llvm_mod)
    llvm_ir, _ = emit_llvm_ir(llvm_mod)
    # Every label block arg should produce a phi instruction
    assert "phi" in llvm_ir, f"No phi in LLVM IR:\n{llvm_ir}"
    # Verify the IR is valid LLVM
    import llvmlite.binding as llvm_binding
    llvm_binding.initialize_native_target()
    mod = llvm_binding.parse_assembly(llvm_ir)
    mod.verify()


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
