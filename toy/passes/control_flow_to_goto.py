"""Lower control_flow dialect to goto dialect.

Converts structured control flow (ForOp) into unstructured labels and branches.
Non-control-flow ops are identity-cloned with remapped operands to stay valid
after block restructuring.

Loop comparisons and increments use algebra ops (not LLVM ops), keeping the
lowering concerns separated.
"""

from __future__ import annotations

from collections.abc import Callable

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra, builtin, control_flow, goto, index
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.function import Function, FunctionOp
from toy.dialects import memory
from dgen.graph import placeholder_block
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dgen.compiler import Compiler

_EMPTY_PACK = PackOp(values=[], type=builtin.List(element_type=Nil()))


def _make_pack(values: list[dgen.Value]) -> PackOp:
    if not values:
        return _EMPTY_PACK
    return PackOp(values=values, type=builtin.List(element_type=values[0].type))


def _chain_before(effects: list[dgen.Op], terminal: dgen.Value) -> dgen.Value:
    result: dgen.Value = terminal
    for effect in reversed(effects):
        result = ChainOp(lhs=effect, rhs=result, type=terminal.type)
    return result


class ControlFlowToGoto(Pass):
    """Lower ForOp to goto labels; identity-clone everything else."""

    def __init__(self) -> None:
        self.loop_counter = 0
        self.value_map: dict[dgen.Value, dgen.Value] = {}
        self._seen: set[dgen.Value] = set()
        self._header_selfs: list[BlockArgument] = []

    def run(self, m: Module, compiler: Compiler[object]) -> Module:
        return Module(
            ops=[
                self._lower_function(op) if isinstance(op, FunctionOp) else op
                for op in m.ops
            ]
        )

    def _lower_function(self, f: FunctionOp) -> FunctionOp:
        self.loop_counter = 0
        self.value_map = {}
        self._seen = set()
        self._header_selfs = []
        for arg in f.body.args:
            self.value_map[arg] = arg
        result = self._lower_ops(f.body.ops, lambda: self._map(f.body.result))
        return FunctionOp(
            name=f.name,
            body=dgen.Block(result=result, args=f.body.args),
            result=f.result,
            type=Function(result=f.result),
        )

    def _map(self, v: dgen.Value) -> dgen.Value:
        return self.value_map.get(v, v)

    def _lower_ops(
        self, ops: list[dgen.Op], return_builder: Callable[[], dgen.Value]
    ) -> dgen.Value:
        effects: list[dgen.Op] = []
        for i, op in enumerate(ops):
            if isinstance(op, control_flow.ForOp):
                if op in self._seen:
                    continue
                self._seen.add(op)
                entry_br, exit_label = self._lower_for(op)
                exit_captures: list[dgen.Value] = [
                    *self._header_selfs,
                    *(self._map(a) for a in op.body.args[1:]),
                ]
                exit_result = self._lower_ops(ops[i + 1 :], return_builder)
                exit_label.body = dgen.Block(result=exit_result, captures=exit_captures)
                self._header_selfs.pop()
                del self.value_map[op]
                return _chain_before(effects, entry_br)
            effect = self._clone_op(op)
            if effect is not None:
                effects.append(effect)
        mapped = return_builder()
        return _chain_before([e for e in effects if e is not mapped], mapped)

    def _clone_op(self, op: dgen.Op) -> dgen.Op | None:
        """Identity-clone an op with remapped operands. Returns side effects."""
        if op in self._seen:
            return None
        self._seen.add(op)

        if isinstance(op, ChainOp):
            new_lhs = self._map(op.lhs)
            self.value_map[op] = new_lhs
            return None

        if isinstance(op, ConstantOp):
            clone = ConstantOp(value=op.value, type=op.type)
            self.value_map[op] = clone
            return None

        if isinstance(op, PackOp):
            clone = PackOp(values=[self._map(v) for v in op.values], type=op.type)
            self.value_map[op] = clone
            return None

        # Generic identity clone: same op type, remapped operands.
        clone = _remap(op, self.value_map)
        if clone is not op:
            self.value_map[op] = clone
        # Side-effecting ops must be chained so they're reachable from
        # block.result and claimed by the entry block (LLVM dominance).
        # AllocOps are effects even though they return non-Nil (Reference).
        if isinstance(op.type, Nil) or isinstance(op, memory.AllocOp):
            return clone
        return None

    def _lower_for(self, op: control_flow.ForOp) -> tuple[goto.BranchOp, goto.LabelOp]:
        """Lower one ForOp to goto header/body/exit labels."""
        lid = self.loop_counter
        self.loop_counter += 1

        header_iv = BlockArgument(name=f"i{lid}", type=index.Index())
        body_iv = BlockArgument(name=f"j{lid}", type=index.Index())
        header_self = BlockArgument(name="self", type=goto.Label())

        # Materialize bounds as constants if they aren't already mapped.
        lo_op = self._map(op.lower_bound)
        if lo_op is op.lower_bound:
            lo_op = ConstantOp(
                value=op.lower_bound.__constant__.to_json(), type=index.Index()
            )
            self.value_map[op.lower_bound] = lo_op
        hi_op = self._map(op.upper_bound)
        if hi_op is op.upper_bound:
            hi_op = ConstantOp(
                value=op.upper_bound.__constant__.to_json(), type=index.Index()
            )
            self.value_map[op.upper_bound] = hi_op

        # Header: compare iv < upper_bound, branch to body or exit.
        cmp_op = algebra.LessThanOp(left=header_iv, right=hi_op, type=index.Index())
        exit_label = goto.LabelOp(name=f"loop_exit{lid}", body=placeholder_block())
        body_label = goto.LabelOp(
            name=f"loop_body{lid}",
            body=dgen.Block(result=dgen.Value(type=Nil()), args=[body_iv]),
        )
        cond_br = goto.ConditionalBranchOp(
            condition=cmp_op,
            true_target=body_label,
            false_target=exit_label,
            true_arguments=_make_pack([header_iv]),
            false_arguments=_EMPTY_PACK,
        )
        header_label = goto.LabelOp(
            name=f"loop_header{lid}",
            body=dgen.Block(result=cond_br, parameters=[header_self], args=[header_iv]),
        )

        # Register header_self so nested blocks can capture it.
        self.value_map[op] = header_self
        self._header_selfs.append(header_self)

        # Body captures: all header_selfs + outer ivars.
        body_captures: list[dgen.Value] = [
            *self._header_selfs,
            *(self._map(a) for a in op.body.args[1:]),
        ]

        # Map loop IV for the body; save/restore outer ivar mappings.
        outer_ivs = list(op.body.args[1:])
        saved = {a: self.value_map[a] for a in outer_ivs if a in self.value_map}
        self.value_map[op.body.args[0]] = body_iv

        def _make_back_br() -> dgen.Value:
            one = ConstantOp(value=1, type=index.Index())
            next_iv = algebra.AddOp(
                left=self.value_map[op.body.args[0]], right=one, type=index.Index()
            )
            return goto.BranchOp(target=self._map(op), arguments=_make_pack([next_iv]))

        body_result = self._lower_ops(op.body.ops, _make_back_br)
        body_label.body = dgen.Block(
            result=body_result, args=[body_iv], captures=body_captures
        )

        for a, v in saved.items():
            self.value_map[a] = v

        # Header captures: enclosing header_selfs (not ours) + outer ivars.
        header_captures: list[dgen.Value] = [
            *self._header_selfs[:-1],
            *(self._map(a) for a in op.body.args[1:]),
        ]
        header_label.body = dgen.Block(
            result=cond_br,
            parameters=[header_self],
            args=[header_iv],
            captures=header_captures,
        )

        entry_br = goto.BranchOp(target=header_label, arguments=_make_pack([lo_op]))
        return entry_br, exit_label


def _remap(op: dgen.Op, vmap: dict[dgen.Value, dgen.Value]) -> dgen.Op:
    """Create a clone of op with operands looked up through vmap.

    Returns the original op unchanged if no operands were remapped.
    """
    mapped_operands: dict[str, dgen.Value] = {}
    changed = False
    for name, val in op.operands:
        if isinstance(val, PackOp):
            new_values = [vmap.get(v, v) for v in val.values]
            if any(n is not o for n, o in zip(new_values, val.values)):
                mapped_operands[name] = PackOp(values=new_values, type=val.type)
                changed = True
            else:
                mapped_operands[name] = val
        else:
            new = vmap.get(val, val)
            mapped_operands[name] = new
            if new is not val:
                changed = True
    if not changed:
        return op
    kwargs: dict[str, dgen.Value | dgen.Block] = {
        "name": op.name,
        "type": op.type,
        **mapped_operands,
    }
    for name, val in op.parameters:
        kwargs[name] = val
    for name, block in op.blocks:
        kwargs[name] = block
    return type(op)(**kwargs)
