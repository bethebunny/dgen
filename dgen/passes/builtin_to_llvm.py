"""Lower builtin dialect ops that don't survive to codegen."""

from __future__ import annotations

import dgen
from dgen.dialects.builtin import TypeOp
from dgen.module import ConstantOp
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import TypeType, type_constant


class BuiltinToLLVM(Pass):
    allow_unregistered_ops = True

    @lowering_for(TypeOp)
    def lower_type(self, op: TypeOp) -> dgen.Value:
        resolved = type_constant(op.value.type)
        return ConstantOp(value=resolved.__constant__.to_json(), type=TypeType())
