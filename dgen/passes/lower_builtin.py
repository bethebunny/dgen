"""Lower builtin-dialect ops that don't survive to later passes.

Currently handles ``builtin.unpack``; a home for future builtin-level
lowerings that produce other dgen dialects (goto, record, ...).

``builtin.unpack`` binds every field of a Tuple to an SSA name in a body
block while preserving dgen's single-result-per-op invariant:

    %r : T = unpack(%t) body(%a: T0, %b: T1, ...):
        <body using %a, %b, ...>

It lowers to a ``goto.region`` that passes the tuple straight through as
``initial_arguments``:

    %r : T = goto.region(%t) body<%_, %r_exit>(%a: T0, %b: T1, ...):
        <body>

The codegen aggregate-bundle phi already extracts each body arg from
the predecessor tuple via ``extractvalue``, so no per-field op is
needed at the dgen level. The synthesized ``%self`` / ``%exit``
parameters satisfy the goto.region shape; ``%self`` is named ``%_``
because back-edges never fire (unpack has no loops). ``%exit`` carries
the body's value via its phi (after ``NormalizeRegionTerminators``)
and gets a unique name for codegen's basic-block label scheme.
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
        exit_name = f"{op.name}_exit" if op.name else "unpack_exit"
        body.parameters = [
            BlockParameter(name="_", type=goto.Label()),
            BlockParameter(name=exit_name, type=goto.Label()),
            *body.parameters,
        ]
        return goto.RegionOp(
            name=op.name,
            initial_arguments=op.tuple,
            type=op.type,
            body=body,
        )
