"""IR-level tests for the autodiff transformation.

Tests the grad(f) → f_grad FunctionOp synthesis at each pipeline stage:
- After lower_grad: symbolic gradient IR with InferredShapeTensor types
- After shape inference: concrete Tensor types propagated through the gradient
- Full pipeline: gradient function lowered to structured IR / LLVM
"""

from dgen.compiler import Compiler, IdentityPass
from dgen.module import Module
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.memory_to_llvm import MemoryToLLVM
from dgen.passes.ndbuffer_to_memory import NDBufferToMemory
from toy.dialects import shape_constant
from toy.dialects.toy import Tensor
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.autodiff import lower_grad
from toy.passes.optimize import ToyOptimize
from toy.passes.shape_inference import ShapeInference
from toy.passes.toy_to_structured import ToyToStructured
from toy.test.helpers import strip_prefix


def _lower_and_grad(source: str) -> Module:
    """Parse → lower → expand grad. No shape inference, no optimization."""
    ir = lower(parse_toy(source))
    func_map = {f.name: f for f in ir.functions if f.name is not None}
    return lower_grad(ir, func_map)


def _lower_grad_and_infer(
    source: str, *, arg_shapes: list[list[int]] | None = None
) -> Module:
    """Parse → lower → expand grad → optimize → shape inference."""
    ir = lower(parse_toy(source))
    if arg_shapes:
        main = ir.functions[0]
        for param, shape in zip(main.body.args, arg_shapes):
            param.type = Tensor(shape=shape_constant(shape))
    func_map = {f.name: f for f in ir.functions if f.name is not None}
    ir = lower_grad(ir, func_map)
    compiler = Compiler(passes=[ToyOptimize(), ShapeInference()], exit=IdentityPass())
    return compiler.run(ir)


_full_compiler = Compiler(
    passes=[
        ToyOptimize(),
        ShapeInference(),
        ToyToStructured(),
        ControlFlowToGoto(),
        NDBufferToMemory(),
        MemoryToLLVM(),
    ],
    exit=IdentityPass(),
)


def _compile_grad(source: str, *, arg_shapes: list[list[int]] | None = None) -> Module:
    """Parse → lower → expand grad → full pipeline (through to LLVM IR)."""
    ir = lower(parse_toy(source))
    if arg_shapes:
        main = ir.functions[0]
        for param, shape in zip(main.body.args, arg_shapes):
            param.type = Tensor(shape=shape_constant(shape))
    func_map = {f.name: f for f in ir.functions if f.name is not None}
    ir = lower_grad(ir, func_map)
    return _full_compiler.run(ir)


# ===----------------------------------------------------------------------=== #
# After lower_grad: symbolic gradient IR
# ===----------------------------------------------------------------------=== #


def test_grad_add_synthesis(ir_snapshot):
    """grad(f) where f(x) = x + x synthesizes f_grad with two adjoint paths."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var y = x + x;
        |   return y;
        | }
    """)
    assert _lower_and_grad(source) == ir_snapshot


def test_grad_mul_synthesis(ir_snapshot):
    """grad(f) where f(x) = x * x synthesizes f_grad with product rule."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var y = x * x;
        |   return y;
        | }
    """)
    assert _lower_and_grad(source) == ir_snapshot


def test_grad_polynomial_synthesis(ir_snapshot):
    """grad(f) where f(x) = x*x + x synthesizes both add and mul adjoint rules."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var y = x * x + x;
        |   return y;
        | }
    """)
    assert _lower_and_grad(source) == ir_snapshot


def test_grad_transpose_synthesis(ir_snapshot):
    """grad through transpose: adjoint of transpose is transpose."""
    source = strip_prefix("""
        | def main() {
        |   var x = [[1, 2, 3], [4, 5, 6]];
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var y = transpose(x);
        |   var z = y * y;
        |   return z;
        | }
    """)
    assert _lower_and_grad(source) == ir_snapshot


def test_grad_constant_synthesis(ir_snapshot):
    """grad(f) where f(x) = x + c: constant has zero adjoint, gradient is fill_like(1, ...)."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var c = [1, 1, 1];
        |   var y = x + c;
        |   return y;
        | }
    """)
    assert _lower_and_grad(source) == ir_snapshot


def test_grad_variable_syntax_synthesis(ir_snapshot):
    """var df = grad(f); df(x) produces the same CallOp(callee=f_grad) IR."""
    source = strip_prefix("""
        | def main(x) {
        |   var df = grad(f);
        |   print(df(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var y = x * x;
        |   return y;
        | }
    """)
    assert _lower_and_grad(source) == ir_snapshot


def test_grad_primal_pruned():
    """Helper function f is removed from the module when only used via grad."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   return x;
        | }
    """)
    ir = _lower_and_grad(source)
    names = [f.name for f in ir.functions]
    assert "f" not in names, "f should be pruned (only used as grad target)"
    assert "f_grad" in names
    assert "main" in names


def test_grad_primal_kept_when_called_directly():
    """f is kept if it's both called directly and used via grad."""
    source = strip_prefix("""
        | def main(x) {
        |   print(f(x));
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   return x + x;
        | }
    """)
    ir = _lower_and_grad(source)
    names = [f.name for f in ir.functions]
    assert "f" in names, "f should be kept (called directly)"
    assert "f_grad" in names


# ===----------------------------------------------------------------------=== #
# After shape inference: concrete types on the gradient function
# ===----------------------------------------------------------------------=== #


def test_grad_add_shape_inference(ir_snapshot):
    """Shape inference propagates [3] through f_grad's body."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   return x + x;
        | }
    """)
    assert _lower_grad_and_infer(source, arg_shapes=[[3]]) == ir_snapshot


def test_grad_mul_shape_inference(ir_snapshot):
    """Shape inference propagates [2, 2] through f_grad's product rule."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   return x * x;
        | }
    """)
    assert _lower_grad_and_infer(source, arg_shapes=[[2, 2]]) == ir_snapshot


# ===----------------------------------------------------------------------=== #
# Full pipeline: gradient function lowered all the way through
# ===----------------------------------------------------------------------=== #


def test_grad_add_full_pipeline(ir_snapshot):
    """f(x) = x + x → full pipeline produces correct LLVM-level IR."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   return x + x;
        | }
    """)
    assert _compile_grad(source, arg_shapes=[[3]]) == ir_snapshot


def test_grad_mul_full_pipeline(ir_snapshot):
    """f(x) = x * x → product rule through full pipeline."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   return x * x;
        | }
    """)
    assert _compile_grad(source, arg_shapes=[[3]]) == ir_snapshot
