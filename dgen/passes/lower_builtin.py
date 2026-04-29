"""Lower builtin-dialect ops that don't survive to later passes.

Currently handles ``builtin.unpack``; a home for future builtin-level
lowerings that produce other dgen dialects (goto, record, ...).

``builtin.unpack`` binds every field of a Tuple to an SSA name in a body
block while preserving dgen's single-result-per-op invariant:

    %r : T = unpack(%t) body(%a: T0, %b: T1, ...):
        <body using %a, %b, ...>

It lowers to a ``goto.region`` that passes the tuple straight through as
``initial_arguments``:

    %r : T = goto.region(%t) body<%_0, %_1>(%a: T0, %b: T1, ...):
        <body>

The codegen aggregate-bundle phi already extracts each body arg from
the predecessor tuple via ``extractvalue``, so no per-field op is
needed at the dgen level. The synthesized ``%self`` / ``%exit``
parameters satisfy the goto.region shape but have no source name —
back-edges never fire (unpack has no loops) and the exit label gets
a tracker-generated name at codegen.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockParameter
from dgen.dialects import builtin, goto
from dgen.passes.pass_ import Pass, lowering_for


class LowerBuiltin(Pass):
    allow_unregistered_ops = True

    @lowering_for(builtin.UnpackOp)
    def lower_unpack(self, op: builtin.UnpackOp) -> dgen.Value | None:
        body = op.body
        body.parameters = [
            BlockParameter(type=goto.Label()),
            BlockParameter(type=goto.Label()),
            *body.parameters,
        ]
        return goto.RegionOp(
            name=op.name,
            initial_arguments=op.tuple,
            type=op.type,
            body=body,
        )
