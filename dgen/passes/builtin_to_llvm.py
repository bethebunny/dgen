"""Lower builtin dialect ops to LLVM dialect ops.

Handles: AddIndexOp, SubtractIndexOp, EqualIndexOp, IfOp, CallOp.
Passes through unchanged: ConstantOp, PackOp, and any LLVM dialect ops.
"""

from __future__ import annotations

import dgen
from dgen.block import BlockArgument, Block
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import FunctionOp, Nil, String
from dgen.graph import placeholder_block
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dgen.compiler import Compiler


_EMPTY_PACK = PackOp(values=[], type=builtin.List(element_type=builtin.Nil()))


def _chain_before(effects: list[dgen.Op], terminal: dgen.Value) -> dgen.Value:
    """Chain effects before terminal so they execute in list order."""
    result: dgen.Value = terminal
    for effect in reversed(effects):
        result = builtin.ChainOp(lhs=effect, rhs=result, type=terminal.type)
    return result


class BuiltinToLLVMLowering(Pass):
    def __init__(self) -> None:
        self.if_counter = 0
        self.value_map: dict[dgen.Value, dgen.Value] = {}

    def run(self, m: Module, compiler: Compiler[object]) -> Module:
        ops = [
            self._lower_function(op) if isinstance(op, FunctionOp) else op
            for op in m.ops
        ]
        return Module(ops=ops)

    def _lower_function(self, f: FunctionOp) -> FunctionOp:
        self.if_counter = 0
        self.value_map = {}

        # First pass: recursively lower builtin ops inside pre-built label bodies.
        visited: set[int] = set()
        for op in f.body.ops:
            if isinstance(op, llvm.LabelOp):
                self._lower_label_bodies(op, visited)

        # Lower entry ops (skip LabelOps — already lowered above).
        non_label_ops = [op for op in f.body.ops if not isinstance(op, llvm.LabelOp)]
        result = self._lower_ops(non_label_ops, f.body.result)
        return FunctionOp(
            name=f.name,
            body=dgen.Block(result=result, args=f.body.args),
            result=f.result,
        )

    def _lower_label_bodies(self, label_op: llvm.LabelOp, visited: set[int]) -> None:
        """Recursively lower builtin ops inside pre-built label body blocks."""
        if id(label_op) in visited:
            return
        visited.add(id(label_op))

        # Recurse into nested labels first.
        for op in label_op.body.ops:
            if isinstance(op, llvm.LabelOp):
                self._lower_label_bodies(op, visited)

        # Lower body ops (skip nested labels — already handled above).
        body_ops = [op for op in label_op.body.ops if not isinstance(op, llvm.LabelOp)]
        result = self._lower_ops(body_ops, label_op.body.result)
        label_op.body = dgen.Block(result=result, args=label_op.body.args)

    def _map(self, old: dgen.Value) -> dgen.Value:
        return self.value_map.get(old, old)

    def _lower_ops(self, ops: list[dgen.Op], return_val: dgen.Value) -> dgen.Value:
        """Lower a sequence of ops, returning the block result with effects chained."""
        effects: list[dgen.Op] = []
        for i, op in enumerate(ops):
            if isinstance(op, builtin.IfOp):
                cond_br, merge_label = self._lower_if(op)
                exit_result = self._lower_ops(ops[i + 1 :], return_val)
                merge_label.body = dgen.Block(
                    result=exit_result, args=merge_label.body.args
                )
                return _chain_before(effects, cond_br)
            effect = self._lower_single_op(op)
            if effect is not None:
                effects.append(effect)
        mapped = self.value_map.get(return_val, return_val)
        return _chain_before([e for e in effects if e is not mapped], mapped)

    def _lower_single_op(self, op: dgen.Op) -> dgen.Op | None:
        """Lower one non-IfOp. Returns the op if it's a side effect, else None."""
        if isinstance(op, builtin.AddIndexOp):
            llvm_op = llvm.AddOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            self.value_map[op] = llvm_op
            return None
        if isinstance(op, builtin.SubtractIndexOp):
            llvm_op = llvm.SubOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            self.value_map[op] = llvm_op
            return None
        if isinstance(op, builtin.EqualIndexOp):
            icmp_op = llvm.IcmpOp(
                pred=String().constant("eq"),
                lhs=self._map(op.lhs),
                rhs=self._map(op.rhs),
            )
            zext_op = llvm.ZextOp(input=icmp_op)
            self.value_map[op] = zext_op
            return None
        if isinstance(op, builtin.CallOp):
            return self._lower_call(op)
        # Pass through: ConstantOp, PackOp, ChainOp, and all LLVM ops.
        return None

    def _lower_if(self, op: builtin.IfOp) -> tuple[llvm.CondBrOp, llvm.LabelOp]:
        if_id = self.if_counter
        self.if_counter += 1

        then_label_op = llvm.LabelOp(name=f"then_{if_id}", body=placeholder_block())
        else_label_op = llvm.LabelOp(name=f"else_{if_id}", body=placeholder_block())

        merge_result_arg = BlockArgument(name=f"merge_val{if_id}", type=op.type)
        merge_label_op = llvm.LabelOp(
            name=f"merge_{if_id}",
            body=dgen.Block(result=merge_result_arg, args=[merge_result_arg]),
        )

        # Convert i64 condition to i1 via icmp ne 0
        zero = ConstantOp(value=0, type=builtin.Index())
        cond_i1 = llvm.IcmpOp(
            pred=String().constant("ne"),
            lhs=self._map(op.cond),
            rhs=zero,
        )
        cond_br = llvm.CondBrOp(
            cond=cond_i1,
            true_target=then_label_op,
            false_target=else_label_op,
            true_args=_EMPTY_PACK,
            false_args=_EMPTY_PACK,
        )

        then_label_op.body = self._lower_branch(
            op.then_body.ops, op.then_body.result, merge_label_op, op.then_body.args
        )
        else_label_op.body = self._lower_branch(
            op.else_body.ops, op.else_body.result, merge_label_op, op.else_body.args
        )

        self.value_map[op] = merge_result_arg
        return cond_br, merge_label_op

    def _lower_branch(
        self,
        ops: list[dgen.Op],
        return_val: dgen.Value,
        merge_label_op: llvm.LabelOp,
        args: list[BlockArgument],
    ) -> Block:
        """Lower branch ops into a Block ending with BrOp to merge_label_op."""
        effects: list[dgen.Op] = []
        for i, op in enumerate(ops):
            if isinstance(op, builtin.IfOp):
                inner_cond_br, inner_merge = self._lower_if(op)
                inner_merge.body = self._lower_branch(
                    ops[i + 1 :], return_val, merge_label_op, inner_merge.body.args
                )
                return Block(result=_chain_before(effects, inner_cond_br), args=args)
            effect = self._lower_single_op(op)
            if effect is not None:
                effects.append(effect)

        branch_result = self._map(return_val)
        if not isinstance(branch_result, Nil):
            result_pack = PackOp(
                values=[branch_result],
                type=builtin.List(element_type=branch_result.type),
            )
            br = llvm.BrOp(target=merge_label_op, args=result_pack)
        else:
            br = llvm.BrOp(target=merge_label_op, args=_EMPTY_PACK)
        return Block(result=_chain_before(effects, br), args=args)

    def _lower_call(self, op: builtin.CallOp) -> llvm.CallOp | None:
        callee_name = op.callee.name
        assert callee_name is not None
        if isinstance(op.args, PackOp):
            mapped_args = [self._map(v) for v in op.args.values]
        else:
            mapped_args = [self._map(op.args)]
        pack = PackOp(values=mapped_args, type=op.args.type)
        llvm_call = llvm.CallOp(
            callee=String().constant(callee_name),
            args=pack,
            type=op.type,
        )
        self.value_map[op] = llvm_call
        if isinstance(op.type, Nil):
            return llvm_call  # side effect: must be chained
        return None  # non-void: reachable via data deps of consumers
