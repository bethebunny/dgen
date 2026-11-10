"""toy's requires clauses are checked: concat and dim_size bound their axis."""

import pytest

from dgen.asm.parser import parse
from dgen.compiler import verify_passes
from dgen.verify import ConstraintError, verify_constraints
from toy.test.helpers import run_toy, strip_prefix

TENSOR = "toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64>"
INFERRED = "toy.InferredShapeTensor<number.Float64>"


def _concat(axis: int, operand: str) -> str:
    return strip_prefix(f"""
        | import function
        | import index
        | import ndbuffer
        | import number
        | import toy
        |
        | %f : function.Function<[{operand}, {operand}], {INFERRED}> = function.function<{INFERRED}>() body(%a: {operand}, %b: {operand}):
        |     %c : {INFERRED} = toy.concat<{axis}>(%a, %b)
    """)


def test_concat_axis_in_range() -> None:
    verify_constraints(parse(_concat(1, TENSOR)), decided=True)


def test_concat_axis_past_rank() -> None:
    with pytest.raises(
        ConstraintError,
        match=r"ConcatOp %c: requires axis < lhs.shape.rank is false "
        r"\(axis = 2, lhs.shape.rank = 2\)",
    ):
        verify_constraints(parse(_concat(2, TENSOR)))


def test_concat_waits_for_shape_inference() -> None:
    """Before shape inference the operand's type has no shape to read."""
    verify_constraints(parse(_concat(5, INFERRED)))
    with pytest.raises(
        ConstraintError,
        match="is undecided: lhs: InferredShapeTensor has no parameter 'shape'",
    ):
        verify_constraints(parse(_concat(5, INFERRED)), decided=True)


def test_program_with_concat_axis_past_rank_is_rejected() -> None:
    with pytest.raises(
        ConstraintError, match=r"is false \(axis = 2, lhs.shape.rank = 2\)"
    ):
        run_toy("""
            def main() {
                var a = [[1, 2, 3], [4, 5, 6]];
                var b = [[7, 8, 9]];
                print(concat(a, b, 2));
                return;
            }
        """)


def test_program_with_dim_size_axis_past_rank_is_rejected() -> None:
    with pytest.raises(
        ConstraintError,
        match=r"DimSizeOp %\w+: requires axis < input.shape.rank is false "
        r"\(axis = 1, input.shape.rank = 1\)",
    ):
        run_toy("""
            def main() {
                var a = [1, 2, 3];
                var n = dim_size(a, 1);
                print(tile(a, n));
                return;
            }
        """)


def test_program_is_rejected_without_verify_passes() -> None:
    """Outside the test suite each pass still checks the ops it visits."""
    token = verify_passes.set(False)
    try:
        with pytest.raises(ConstraintError, match=r"axis < lhs.shape.rank is false"):
            run_toy("""
                def main() {
                    var a = [[1, 2, 3], [4, 5, 6]];
                    print(concat(a, a, 3));
                    return;
                }
            """)
    finally:
        verify_passes.reset(token)
