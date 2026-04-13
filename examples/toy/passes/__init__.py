"""Toy dialect passes."""

from __future__ import annotations

import dgen
from dgen.passes.compiler import Compiler, IdentityPass
from toy.passes.optimize import ToyOptimize
from toy.passes.shape_inference import ShapeInference
from toy.passes.toy_to_structured import ToyToStructured


def lower_toy() -> Compiler[dgen.Value]:
    """Toy dialect lowerings: optimize, shape inference, lower to structured."""
    return Compiler(
        passes=[ToyOptimize(), ShapeInference(), ToyToStructured()],
        exit=IdentityPass(),
    )
