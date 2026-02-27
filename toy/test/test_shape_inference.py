"""Ch4 tests: Shape inference pass."""

from dgen import asm
from dgen.asm.parser import parse_module
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.shape_inference import infer_shapes
from toy.test.helpers import strip_prefix


def compile_and_infer(source: str) -> str:
    ast = parse_toy(source)
    ir = lower(ast)
    ir = infer_shapes(ir)
    return asm.format(ir)


def test_transpose():
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(a);
        |   print(b);
        |   return;
        | }
    """)
    result = compile_and_infer(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 2], f64> = toy.transpose(%0)
        |     %2 : () = toy.print(%1)
        |     %3 : () = return(())
    """)
    assert result == expected


def test_mul():
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile_and_infer(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2], f64> = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor<[2, 2], f64> = [5.0, 6.0, 7.0, 8.0]
        |     %2 : toy.Tensor<[2, 2], f64> = toy.mul(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %4 : () = return(())
    """)
    assert result == expected


def test_3d_add():
    source = strip_prefix("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a + b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile_and_infer(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], f64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<[2, 2, 2], f64> = toy.add(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %4 : () = return(())
    """)
    assert result == expected


def test_3d_mul():
    source = strip_prefix("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile_and_infer(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 2, 2], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor<[2, 2, 2], f64> = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<[2, 2, 2], f64> = toy.mul(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %4 : () = return(())
    """)
    assert result == expected


def test_generic_call():
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
    result = compile_and_infer(source)
    expected = strip_prefix("""
        | import toy
        |
        | %multiply_transpose = function (%a: toy.Tensor<[2, 3], f64>, %b: toy.Tensor<[2, 3], f64>) -> toy.Tensor<[3, 2], f64>:
        |     %0 : toy.Tensor<[3, 2], f64> = toy.transpose(%a)
        |     %1 : toy.Tensor<[3, 2], f64> = toy.transpose(%b)
        |     %2 : toy.Tensor<[3, 2], f64> = toy.mul(%0, %1)
        |     %3 : () = return(%2)
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %2 : toy.Tensor<[3, 2], f64> = toy.generic_call("multiply_transpose", [%0, %1])
        |     %3 : () = toy.print(%2)
        |     %4 : () = return(())
    """)
    assert result == expected


def test_concat():
    """Concat shape is computed from input shapes: [2,3] concat [3,3] axis=0 -> [5,3]."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.InferredShapeTensor<f64> = toy.concat<0>(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    result = infer_shapes(module)
    out = asm.format(result)
    expected = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[3, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.Tensor<[5, 3], f64> = toy.concat<0>(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return(())
    """)
    assert out == expected


def test_concat_axis1():
    """Concat along axis 1: [2,3] concat [2,5] -> [2,8]."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[2, 5], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        |     %2 : toy.InferredShapeTensor<f64> = toy.concat<1>(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    result = infer_shapes(module)
    out = asm.format(result)
    expected = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[2, 5], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        |     %2 : toy.Tensor<[2, 8], f64> = toy.concat<1>(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return(())
    """)
    assert out == expected


def test_tile_with_constant_count():
    """Tile where count is a constant — shape inference peeks through the constant."""
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor<[3], f64> = [1.0, 2.0, 3.0]
        |     %1 : index = 4
        |     %2 : toy.InferredShapeTensor<f64> = toy.tile<%1>(%0)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    result = infer_shapes(module)
    out = asm.format(result)
    expected = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor<[3], f64> = [1.0, 2.0, 3.0]
        |     %1 : index = 4
        |     %2 : toy.Tensor<[4, 3], f64> = toy.tile<%1>(%0)
        |     %3 : () = toy.print(%2)
        |     %_ : () = return(())
    """)
    assert out == expected


def test_tile_with_computed_count():
    """Tile where count is add_index(2, 2) — shape inference CANNOT resolve this.

    This is the staging boundary: the shape depends on a value that requires
    evaluation. The InferredShapeTensor remains unresolved.
    """
    ir = strip_prefix("""
        | import toy
        |
        | %f = function () -> ():
        |     %0 : toy.Tensor<[3], f64> = [1.0, 2.0, 3.0]
        |     %1 : index = 2
        |     %2 : index = 2
        |     %3 : index = add_index(%1, %2)
        |     %4 : toy.InferredShapeTensor<f64> = toy.tile<%0>(%3)
        |     %5 : () = toy.print(%4)
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    result = infer_shapes(module)
    out = asm.format(result)
    # Shape inference cannot resolve %4 — it stays InferredShapeTensor
    # because the count (%3) is not a constant, it's a computed value.
    # Resolving this requires a staging evaluator.
    assert "toy.InferredShapeTensor<f64> = toy.tile" in out


def test_full_tutorial_example():
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
    result = compile_and_infer(source)
    expected = strip_prefix("""
        | import toy
        |
        | %multiply_transpose = function (%a: toy.Tensor<[2, 3], f64>, %b: toy.Tensor<[2, 3], f64>) -> toy.Tensor<[3, 2], f64>:
        |     %0 : toy.Tensor<[3, 2], f64> = toy.transpose(%a)
        |     %1 : toy.Tensor<[3, 2], f64> = toy.transpose(%b)
        |     %2 : toy.Tensor<[3, 2], f64> = toy.mul(%0, %1)
        |     %3 : () = return(%2)
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor<[2, 3], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor<[2, 3], f64> = toy.reshape(%0)
        |     %2 : toy.Tensor<[6], f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor<[2, 3], f64> = toy.reshape(%2)
        |     %4 : toy.Tensor<[3, 2], f64> = toy.generic_call("multiply_transpose", [%1, %3])
        |     %5 : toy.Tensor<[3, 2], f64> = toy.generic_call("multiply_transpose", [%3, %1])
        |     %6 : () = toy.print(%5)
        |     %7 : () = return(())
    """)
    assert result == expected
