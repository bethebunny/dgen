"""Diff dialect: automatic differentiation ops for the Toy language.

Defines a GradOp that references a function and produces its gradient.
The autodiff pass lowers GradOp into concrete Toy ops.
"""

from __future__ import annotations

from dataclasses import dataclass

from dgen import Dialect, Op, Type, Value
from dgen.dialects.function import Function

diff = Dialect("diff")


@diff.op("grad")
@dataclass(eq=False, kw_only=True)
class GradOp(Op):
    """Evaluate the gradient of a function at given arguments.

    callee: reference to the function to differentiate.
    arguments: the point at which to evaluate the gradient.
    The result is the gradient tensor(s).
    """

    callee: Value[Function]
    arguments: Value
    type: Type
