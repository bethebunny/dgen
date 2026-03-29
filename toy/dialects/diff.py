"""Diff dialect: automatic differentiation ops for the Toy language.

Defines a GradOp that wraps a function reference and produces its gradient
function. The autodiff pass lowers CallOp(callee=GradOp) into concrete
reverse-mode AD computation.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from dgen import Dialect, Op, Type, Value
from dgen.dialects.function import Function

diff = Dialect("diff")


@diff.op("grad")
@dataclass(eq=False, kw_only=True)
class GradOp(Op):
    """The gradient function of a given function.

    callee: reference to the function to differentiate.
    type: Function — grad(f) produces a function value.

    callee is NOT exposed as a __params__ field — it is an opaque reference
    that the autodiff pass resolves. This prevents the staging pass from
    trying to JIT-evaluate it.
    """

    callee: Value[Function]
    type: Type

    @property
    def dependencies(self) -> Iterator[Value]:
        yield self.type
        yield self.callee
