"""Diff dialect: automatic differentiation ops for the Toy language.

GradOp takes a Function and produces a Function — the gradient.
The Autodiff pass synthesizes the gradient function body.
"""

from __future__ import annotations

from dataclasses import dataclass

from dgen import Dialect, Op, Type, Value
from dgen.dialects.function import Function

diff = Dialect("diff")


@diff.op("grad")
@dataclass(eq=False, kw_only=True)
class GradOp(Op):
    """The gradient of a function. Takes a Function, produces a Function."""

    function: Value
    type: Type
