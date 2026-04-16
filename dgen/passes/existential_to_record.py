"""Lower existential dialect ops to record + memory ops.

``existential.pack(value)`` becomes:

    %witness    = builtin.type(value)
    %inner      = record.pack([%witness, %value])   : Tuple<[Type, T]>
    %result     = heap_box(%inner)                   : Reference<Tuple<[Type, T]>>

``existential.unpack(box)`` dereferences the pointer and extracts the
value field from the inner record:

    %inner      = memory.load(box, box)              : Tuple<[Type, T]>
    %result     = record.get<1>(%inner)              : T
"""

from __future__ import annotations

import dgen
from dgen.builtins import pack
from dgen.dialects import existential, memory, record
from dgen.dialects.builtin import Tuple, TypeOp
from dgen.dialects.index import Index
from dgen.passes.pass_ import Pass, lowering_for
from dgen.passes.support.memory import heap_box
from dgen.type import TypeType, constant


class ExistentialToRecord(Pass):
    allow_unregistered_ops = True

    @lowering_for(existential.PackOp)
    def lower_pack(self, op: existential.PackOp) -> dgen.Value | None:
        value_type = constant(op.value.type)
        assert isinstance(value_type, dgen.Type)

        witness = TypeOp(value=op.value, type=TypeType())
        inner_type = Tuple(types=pack([TypeType(), value_type]))
        inner = record.PackOp(values=pack([witness, op.value]), type=inner_type)
        return heap_box(inner)

    @lowering_for(existential.UnpackOp)
    def lower_unpack(self, op: existential.UnpackOp) -> dgen.Value | None:
        witness_type = op.type
        assert isinstance(witness_type, dgen.Type)

        inner_type = Tuple(types=pack([TypeType(), witness_type]))
        inner = memory.LoadOp(mem=op.box, ptr=op.box, type=inner_type)
        return record.GetOp(index=Index().constant(1), record=inner, type=witness_type)
