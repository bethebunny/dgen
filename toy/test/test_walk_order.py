"""Pass results don't depend on the walk order (IR core R5)."""

import pytest

from dgen.compiler import Compiler, IdentityPass
from dgen.graph import RandomOrder, random_order
from dgen.ir_equiv import graph_equivalent
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.memory_to_llvm import MemoryToLLVM
from dgen.passes.pass_ import Pass
from dgen.passes.ndbuffer_to_memory import NDBufferToMemory
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.optimize import ToyOptimize
from toy.passes.shape_inference import ShapeInference
from toy.passes.toy_to_structured import ToyToStructured

PASSES: list[Pass] = [
    ToyOptimize(),
    ShapeInference(),
    ToyToStructured(),
    ControlFlowToGoto(),
    NDBufferToMemory(),
    MemoryToLLVM(),
]

TRANSPOSE = """
def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b = transpose(a);
  print(b);
  return;
}
"""


# These set their own orders, so --walk-seeds leaves them alone.
@pytest.mark.canonical_order
@pytest.mark.parametrize("seed", range(3))
def test_passes_independent_of_walk_order(seed: int):
    compiler = Compiler(PASSES, IdentityPass())
    expected = compiler.run(lower(parse_toy(TRANSPOSE)))
    token = random_order.set(RandomOrder(seed))
    try:
        actual = compiler.run(lower(parse_toy(TRANSPOSE)))
    finally:
        random_order.reset(token)
    assert graph_equivalent(actual, expected)
