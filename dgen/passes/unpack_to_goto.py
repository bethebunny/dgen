"""Lower ``builtin.unpack`` to ``goto.region``.

``unpack`` is the way to bind every field of a Tuple to an SSA name while
preserving dgen's single-result-per-op invariant. Its body block has one
arg per Tuple element; the lowering extracts each element via
``record.get<i>`` and feeds them into the region as initial arguments.

    %r : T = builtin.unpack(%t) body(%a: T0, %b: T1, ...):
        <body using %a, %b, ...>

becomes:

    %e0 : T0 = record.get<0>(%t)
    %e1 : T1 = record.get<1>(%t)
    ...
    %r : T = goto.region([%e0, %e1, ...]) body<%self, %exit>(%a: T0, %b: T1, ...):
        <body>

The added ``%self`` / ``%exit`` parameters are required by the goto.region
shape; back-edges never fire (unpack has no loops), and ``%exit`` is wired
up by ``NormalizeRegionTerminators`` as it would for any region.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockParameter
from dgen.builtins import pack
from dgen.dialects import builtin, goto, record
from dgen.dialects.index import Index
from dgen.passes.pass_ import Pass, lowering_for


class UnpackToGoto(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._counter = 0

    @lowering_for(builtin.UnpackOp)
    def lower_unpack(self, op: builtin.UnpackOp) -> dgen.Value | None:
        uid = self._counter
        self._counter += 1

        extracted: list[dgen.Value] = []
        for i, arg in enumerate(op.body.args):
            assert isinstance(arg.type, dgen.Type)
            extracted.append(
                record.GetOp(
                    index=Index().constant(i),
                    record=op.tuple,
                    type=arg.type,
                )
            )

        self_param = BlockParameter(name="self", type=goto.Label())
        exit_param = BlockParameter(name=f"unpack_exit{uid}", type=goto.Label())
        op.body.parameters = [self_param, exit_param, *op.body.parameters]

        return goto.RegionOp(
            name=op.name or f"unpack{uid}",
            initial_arguments=pack(extracted),
            type=op.type,
            body=op.body,
        )
