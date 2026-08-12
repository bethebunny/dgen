"""Lower memory.destroy through the origin's attached destructor.

An origin's type carries its destructor, an ordinary function value set
by memory.attach. destroy applies it. A destructor that is a function
literal is inlined at the destroy site, and any other function value
becomes an indirect call. An origin whose type carries no destructor
(the default) discharges with no runtime action.

Destructor bodies capture the values their cleanup needs, so inlining
adds those captures to each block between their binding and the destroy
site. After lowering, attach ops collapse to their origin operand and
origin types drop their destructor parameter, which unlinks the
destructor functions from the graph.

verify_preconditions checks that origin-threading ops (attach, store,
load) annotate their results with the same origin type they consume.
The spec language cannot yet declare operand-dependent result types
(see TODO.md), so the annotation is checked rather than derived.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dgen
from dgen.block import Block
from dgen.builtins import PackOp, pack
from dgen.dialects import function, memory
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.function import FunctionOp
from dgen.ir.traversal import all_blocks, all_values
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import Constant, constant

if TYPE_CHECKING:
    from dgen.passes.compiler import Compiler


class OriginThreadTypeError(Exception):
    """An origin-threading op's annotated result type does not match the
    origin type it consumes."""


class SharedDestructorError(Exception):
    """A destructor function literal is attached to more than one
    destroyed origin. Inlining mutates the body, so each literal can be
    spliced once. Sharing needs a block copy utility."""


def _destructor(t: memory.Origin) -> dgen.Value:
    """*t*'s destructor with constant wrapping resolved. A type parsed
    from a fully constant annotation (an origin inside a constant tuple
    type) wraps its parameters in Constants."""
    d = t.destructor
    if isinstance(d, Constant):
        resolved = constant(d)
        if isinstance(resolved, dgen.Type):
            return resolved
    return d


def _same_origin_type(a: dgen.Value, b: dgen.Value) -> bool:
    if not (isinstance(a, memory.Origin) and isinstance(b, memory.Origin)):
        return False
    da, db = _destructor(a), _destructor(b)
    if isinstance(da, Nil) and isinstance(db, Nil):
        return True
    return da is db


def _check_threading(op: dgen.Op, result_type: dgen.Value) -> None:
    assert isinstance(op, (memory.StoreOp, memory.LoadOp))
    if not _same_origin_type(result_type, op.origin.type):
        raise OriginThreadTypeError(
            f"{type(op).__name__} %{op.name} consumes an origin typed "
            f"{op.origin.type.format_asm()} but its result is annotated "
            f"{result_type.format_asm()}. Origin-threading ops must "
            f"annotate their results with the origin type they consume."
        )


class LowerDestroy(Pass):
    allow_unregistered_ops = True

    def __init__(self) -> None:
        self._spliced: set[FunctionOp] = set()
        self._block_stack: list[Block] = []

    def verify_preconditions(self, value: dgen.Value) -> None:
        super().verify_preconditions(value)
        for v in all_values(value):
            if isinstance(v, (memory.StoreOp, memory.LoadOp)):
                _check_threading(v, self._origin_result_type(v))
            elif isinstance(v, memory.AttachOp) and not (
                isinstance(v.type, memory.Origin)
                and _destructor(v.type) is v.destructor
            ):
                raise OriginThreadTypeError(
                    f"attach %{v.name} attaches %{v.destructor.name} but "
                    f"its result is annotated "
                    f"{v.type.format_asm()}. The result annotation must "
                    f"carry the attached destructor."
                )

    @staticmethod
    def _origin_result_type(op: memory.StoreOp | memory.LoadOp) -> dgen.Value:
        """The origin component of *op*'s annotated result type. store
        returns an origin directly, load returns Tuple<[T, Origin]>."""
        if isinstance(op, memory.StoreOp):
            return op.type
        elements = op.type.types
        if isinstance(elements, PackOp):
            return elements.values[1]
        resolved = constant(elements)
        assert isinstance(resolved, list)
        return resolved[1]

    def run(self, value: dgen.Value, compiler: Compiler[object]) -> dgen.Value:
        self._spliced = set()
        self._block_stack = []
        result = super().run(value, compiler)
        # Unlink the destructor bookkeeping. Attach ops collapse to
        # their origin operand and destructor-carrying origin types drop
        # back to the bare Origin, so destructor function literals stop
        # being reachable and codegen never sees them.
        for v in [v for v in all_values(result) if isinstance(v, memory.AttachOp)]:
            result.replace_uses_of(v, v.origin)
        bare = memory.Origin()
        for t in [
            t
            for t in all_values(result)
            if isinstance(t, memory.Origin) and not isinstance(t.destructor, Nil)
        ]:
            result.replace_uses_of(t, bare)
        # A spliced destructor's only remaining references are capture
        # declarations that existed for the type edge. Drop them so the
        # vestigial function value falls out of the graph.
        for block in all_blocks(result):
            if any(c in self._spliced for c in block.captures):
                block.captures = [c for c in block.captures if c not in self._spliced]
        return result

    def _lower_block(self, block: Block) -> None:
        self._block_stack.append(block)
        super()._lower_block(block)
        self._block_stack.pop()

    @lowering_for(memory.DestroyOp)
    def lower_destroy(self, op: memory.DestroyOp) -> dgen.Value | None:
        origin_type = op.origin.type
        assert isinstance(origin_type, memory.Origin)
        destructor = _destructor(origin_type)
        if isinstance(destructor, Nil):
            return ChainOp(lhs=Nil().constant(None), rhs=op.origin, type=Nil())
        if isinstance(destructor, FunctionOp):
            return self._inline_destructor(destructor, op)
        # A runtime function value. Emit an indirect call and leave the
        # origin argument to codegen's zero-sized handling.
        return function.CallOp(
            callee=destructor, arguments=pack([op.origin]), type=Nil()
        )

    def _inline_destructor(
        self, destructor: FunctionOp, op: memory.DestroyOp
    ) -> dgen.Value:
        if destructor in self._spliced:
            raise SharedDestructorError(
                f"destructor %{destructor.name} is applied by more than one "
                f"destroy. Each destructor function literal supports a "
                f"single destroy site."
            )
        self._spliced.add(destructor)
        # Re-type the consumed origin to the bare Origin first. Its type
        # references the destructor, and once the body is spliced the
        # destructor's body references the origin, which would otherwise
        # close a use-def cycle through the type edge.
        op.origin.type = memory.Origin()
        body = destructor.body
        (origin_arg,) = body.args
        body.replace_uses_of(origin_arg, op.origin)
        for cap in body.captures:
            self._thread_capture(cap)
        # Give the vestigial function a fresh empty body. The spliced
        # ops now belong to the destroy site's block, and sharing them
        # from here would put a cycle through the attach's operand edge.
        destructor.body = Block(result=Nil().constant(None))
        return body.result

    def _thread_capture(self, cap: dgen.Value) -> None:
        """Add *cap* to the capture list of each block between its
        binding block and the destroy site."""
        bound_at = None
        for i, block in enumerate(self._block_stack):
            if (
                cap in block.args
                or cap in block.parameters
                or cap in block.captures
                or (isinstance(cap, dgen.Op) and cap in set(block.values))
            ):
                bound_at = i
        if bound_at is None:
            # Not bound anywhere in the stack. Closed-block verification
            # reports this with a precise error.
            return
        for block in self._block_stack[bound_at + 1 :]:
            if cap not in block.captures:
                block.captures = [*block.captures, cap]
