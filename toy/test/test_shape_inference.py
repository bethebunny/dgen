"""Ch4 tests: Shape inference pass."""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.compiler import Compiler, IdentityPass
from dgen.module import Module
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.shape_inference import ShapeInference
from toy.test.helpers import strip_prefix

_compiler = Compiler([], IdentityPass())


def infer_shapes(m: Module) -> Module:
    return ShapeInference().run(m, _compiler)


def compile_and_infer(source: str) -> Module:
    ast = parse_toy(source)
    ir = lower(ast)
    return infer_shapes(ir)


def test_transpose(ir_snapshot):
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(a);
        |   print(b);
        |   return;
        | }
    """)
    assert compile_and_infer(source) == ir_snapshot


def test_mul(ir_snapshot):
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    assert compile_and_infer(source) == ir_snapshot


def test_3d_add(ir_snapshot):
    source = strip_prefix("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a + b;
        |   print(c);
        |   return;
        | }
    """)
    assert compile_and_infer(source) == ir_snapshot


def test_3d_mul(ir_snapshot):
    source = strip_prefix("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    assert compile_and_infer(source) == ir_snapshot


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
    assert compile_and_infer(source) == ir_snapshot


def test_concat(ir_snapshot):
    """Concat shape is computed from input shapes: [2,3] concat [3,3] axis=0 -> [5,3]."""
    ir = strip_prefix("""
        | import function
        | import toy
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<memory.Shape<2>([3, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.InferredShapeTensor<number.Float64> = toy.concat<0>(%0, %1)
        |     %3 : Nil = toy.print(%2)
    """)
    module = parse_module(ir)
    assert infer_shapes(module) == ir_snapshot


def test_concat_axis1(ir_snapshot):
    """Concat along axis 1: [2,3] concat [2,5] -> [2,8]."""
    ir = strip_prefix("""
        | import function
        | import toy
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<memory.Shape<2>([2, 5]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        |     %2 : toy.InferredShapeTensor<number.Float64> = toy.concat<1>(%0, %1)
        |     %3 : Nil = toy.print(%2)
    """)
    module = parse_module(ir)
    assert infer_shapes(module) == ir_snapshot


def test_tile_with_constant_count(ir_snapshot):
    """Tile where count is a constant — shape inference peeks through the constant."""
    ir = strip_prefix("""
        | import function
        | import toy
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<1>([3]), number.Float64> = [1.0, 2.0, 3.0]
        |     %1 : index.Index = 4
        |     %2 : toy.InferredShapeTensor<number.Float64> = toy.tile<%1>(%0)
        |     %3 : Nil = toy.print(%2)
    """)
    module = parse_module(ir)
    assert infer_shapes(module) == ir_snapshot


def test_tile_with_computed_count():
    """Tile where count is index.add(2, 2) — shape inference CANNOT resolve this.

    This is the staging boundary: the shape depends on a value that requires
    evaluation. The InferredShapeTensor remains unresolved.
    """
    ir = strip_prefix("""
        | import algebra
        | import function
        | import toy
        | import index
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %0 : toy.Tensor<memory.Shape<1>([3]), number.Float64> = [1.0, 2.0, 3.0]
        |     %1 : index.Index = 2
        |     %2 : index.Index = 2
        |     %3 : index.Index = algebra.add(%1, %2)
        |     %4 : toy.InferredShapeTensor<number.Float64> = toy.tile<%3>(%0)
        |     %5 : Nil = toy.print(%4)
    """)
    module = parse_module(ir)
    result = infer_shapes(module)
    out = asm.format(result)
    # Shape inference cannot resolve %4 — it stays InferredShapeTensor
    # because the count (%3) is not a constant, it's a computed value.
    # Resolving this requires a staging evaluator.
    assert "toy.InferredShapeTensor<number.Float64> = toy.tile" in out


def test_full_tutorial_example(ir_snapshot):
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
    assert compile_and_infer(source) == ir_snapshot
