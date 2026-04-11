"""Lower builtin dialect ops that don't survive to codegen."""

from __future__ import annotations

import dgen
from dgen.dialects.builtin import TypeOp
from dgen.passes.pass_ import Pass, lowering_for
from dgen.builtins import ConstantOp
from dgen.type import TypeType, constant


class BuiltinToLLVM(Pass):
    allow_unregistered_ops = True

    @lowering_for(TypeOp)
    def lower_type(self, op: TypeOp) -> dgen.Value:
        return ConstantOp.from_constant(TypeType().constant(constant(op.value.type)))
