"""Lower builtin-dialect ops that don't survive to later passes.

Currently handles ``builtin.unpack``; a home for future builtin-level
lowerings that produce other dgen dialects (goto, record, ...).

``builtin.unpack`` binds every field of a Tuple to an SSA name in a body
block while preserving dgen's single-result-per-op invariant:

    %r : T = unpack(%t) body(%a: T0, %b: T1, ...):
        <body using %a, %b, ...>

It lowers to a ``goto.region`` whose body captures the tuple directly and
extracts each named field with ``record.get<i>``:

    %r : T = goto.region([]) body<%_, %r_exit>() captures(%t):
        %a : T0 = record.get<0>(%t)
        %b : T1 = record.get<1>(%t)
        <body>

The synthesized ``%self`` / ``%exit`` parameters satisfy the goto.region
shape; ``%self`` is named ``%_`` because back-edges never fire (unpack
has no loops). ``%exit`` carries the body's value via its phi (after
``NormalizeRegionTerminators``) and gets a unique name for codegen's
basic-block label scheme.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockParameter
from dgen.builtins import pack
from dgen.dialects import builtin, goto, record
from dgen.dialects.index import Index
from dgen.passes.pass_ import Pass, lowering_for


class LowerBuiltin(Pass):
    allow_unregistered_ops = True

    @lowering_for(builtin.UnpackOp)
    def lower_unpack(self, op: builtin.UnpackOp) -> dgen.Value | None:
        body = op.body
        for i, arg in enumerate(list(body.args)):
            assert isinstance(arg.type, dgen.Type)
            getter = record.GetOp(
                name=arg.name,
                index=Index().constant(i),
                record=op.tuple,
                type=arg.type,
            )
            body.replace_uses_of(arg, getter)
        body.args = []
        if op.tuple not in body.captures:
            body.captures.append(op.tuple)
        exit_name = f"{op.name}_exit" if op.name else "unpack_exit"
        body.parameters = [
            BlockParameter(name="_", type=goto.Label()),
            BlockParameter(name=exit_name, type=goto.Label()),
            *body.parameters,
        ]
        return goto.RegionOp(
            name=op.name,
            initial_arguments=pack([]),
            type=op.type,
            body=body,
        )
