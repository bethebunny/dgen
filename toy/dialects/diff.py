"""Diff dialect: automatic differentiation ops for the Toy language.

Defines a GradOp that wraps a function reference and produces its gradient
function. The ``lower_grad`` transformation expands every GradOp into a
synthesized ``FunctionOp`` before the module enters compilation/staging.
"""

from __future__ import annotations

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

    GradOps are expanded into real FunctionOps by ``lower_grad`` before
    the module enters the compiler pipeline. They never reach staging.
    """

    callee: Value[Function]
    type: Type
