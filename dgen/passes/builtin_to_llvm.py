"""Lower builtin dialect ops to LLVM dialect ops.

Handles: AddIndexOp, SubtractIndexOp, EqualIndexOp, IfOp, CallOp.
Passes through unchanged: ConstantOp, PackOp, and any LLVM dialect ops.
"""

from __future__ import annotations

from collections.abc import Iterator

import dgen
from dgen.block import BlockArgument
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import FunctionOp, Nil, String
from dgen.graph import chain_body, group_into_blocks, placeholder_block
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass


_EMPTY_PACK = PackOp(values=[], type=builtin.List(element_type=builtin.Nil()))


class BuiltinToLLVMLowering(Pass):
    def __init__(self) -> None:
        self.if_counter = 0
        self.value_map: dict[dgen.Value, dgen.Value] = {}
        self.current_label: dgen.Value = dgen.Value(name="entry", type=llvm.Label())

    def run(self, m: Module) -> Module:
        functions = [self._lower_function(f) for f in m.functions]
        return Module(functions=functions)

    def _lower_function(self, f: FunctionOp) -> FunctionOp:
        self.if_counter = 0
        self.value_map = {}
        self.current_label = dgen.Value(name="entry", type=llvm.Label())

        # Recursively lower pre-built label bodies (from affine_to_llvm),
        # then lower entry ops. _lower_if may yield new labels into the flat
        # stream, so we still need group_into_blocks afterwards.
        visited: set[int] = set()
        for op in f.body.ops:
            if isinstance(op, llvm.LabelOp):
                self._lower_label_bodies(op, visited)

        flat_ops: list[dgen.Op] = []
        for op in f.body.ops:
            if isinstance(op, llvm.LabelOp):
                continue  # pre-built labels are already lowered
            flat_ops.extend(self.lower_op(op))

        # Group at label boundaries (new labels from _lower_if)
        entry_ops, label_groups = group_into_blocks(flat_ops)
        for label_op, body_ops in label_groups:
            label_op.body = dgen.Block(
                result=chain_body(body_ops), args=label_op.body.args
            )

        if entry_ops:
            new_result = chain_body(entry_ops)
        else:
            new_result = self.value_map.get(f.body.result, f.body.result)
        return FunctionOp(
            name=f.name,
            body=dgen.Block(result=new_result, args=f.body.args),
            result=f.result,
        )

    def _lower_label_bodies(self, label_op: llvm.LabelOp, visited: set[int]) -> None:
        """Recursively lower builtin ops inside pre-built label body blocks."""
        if id(label_op) in visited:
            return
        visited.add(id(label_op))

        # First, recurse into any nested labels found in this body
        for op in label_op.body.ops:
            if isinstance(op, llvm.LabelOp):
                self._lower_label_bodies(op, visited)

        # Lower body ops, rebuilding the block. _lower_if may yield new labels
        # so we need group_into_blocks here too.
        body_flat: list[dgen.Op] = []
        for op in label_op.body.ops:
            if isinstance(op, llvm.LabelOp):
                continue  # nested labels already lowered
            body_flat.extend(self.lower_op(op))

        body_ops, body_label_groups = group_into_blocks(body_flat)
        for lbl, lbl_body_ops in body_label_groups:
            lbl.body = dgen.Block(result=chain_body(lbl_body_ops), args=lbl.body.args)
        label_op.body = dgen.Block(result=chain_body(body_ops), args=label_op.body.args)

    def _map(self, old: dgen.Value) -> dgen.Value:
        return self.value_map.get(old, old)

    def lower_op(self, op: dgen.Op) -> Iterator[dgen.Op]:
        if isinstance(op, builtin.AddIndexOp):
            llvm_op = llvm.AddOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            yield llvm_op
            self.value_map[op] = llvm_op
        elif isinstance(op, builtin.SubtractIndexOp):
            llvm_op = llvm.SubOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            yield llvm_op
            self.value_map[op] = llvm_op
        elif isinstance(op, builtin.EqualIndexOp):
            icmp_op = llvm.IcmpOp(
                pred=String().constant("eq"),
                lhs=self._map(op.lhs),
                rhs=self._map(op.rhs),
            )
            yield icmp_op
            zext_op = llvm.ZextOp(input=icmp_op)
            yield zext_op
            self.value_map[op] = zext_op
        elif isinstance(op, builtin.IfOp):
            yield from self._lower_if(op)
        elif isinstance(op, builtin.CallOp):
            yield from self._lower_call(op)
        else:
            # Pass through ConstantOp, PackOp, LLVM ops, etc. unchanged
            yield op

    def _lower_if(self, op: builtin.IfOp) -> Iterator[dgen.Op]:
        if_id = self.if_counter
        self.if_counter += 1

        then_label_op = llvm.LabelOp(name=f"then_{if_id}", body=placeholder_block())
        else_label_op = llvm.LabelOp(name=f"else_{if_id}", body=placeholder_block())

        # Merge label has a block arg for the if/else result value
        merge_result_arg = BlockArgument(name=f"merge_val{if_id}", type=op.type)
        merge_label_op = llvm.LabelOp(
            name=f"merge_{if_id}",
            body=dgen.Block(
                result=dgen.Value(type=builtin.Nil()),
                args=[merge_result_arg],
            ),
        )

        # Convert i64 condition to i1 via icmp ne 0
        zero = ConstantOp(value=0, type=builtin.Index())
        yield zero
        cond_i1 = llvm.IcmpOp(
            pred=String().constant("ne"),
            lhs=self._map(op.cond),
            rhs=zero,
        )
        yield cond_i1
        yield llvm.CondBrOp(
            cond=cond_i1,
            true_target=then_label_op,
            false_target=else_label_op,
            true_args=_EMPTY_PACK,
            false_args=_EMPTY_PACK,
        )

        # Then block — lower body ops; block result is the branch value
        yield then_label_op
        self.current_label = then_label_op
        for child in op.then_body.ops:
            yield from self.lower_op(child)
        then_result = self._map(op.then_body.result)
        if not isinstance(then_result, Nil):
            result_pack = PackOp(
                values=[then_result],
                type=builtin.List(element_type=then_result.type),
            )
            yield result_pack
            yield llvm.BrOp(target=merge_label_op, args=result_pack)
        else:
            yield llvm.BrOp(target=merge_label_op, args=_EMPTY_PACK)

        # Else block — same pattern
        yield else_label_op
        self.current_label = else_label_op
        for child in op.else_body.ops:
            yield from self.lower_op(child)
        else_result = self._map(op.else_body.result)
        if not isinstance(else_result, Nil):
            result_pack = PackOp(
                values=[else_result],
                type=builtin.List(element_type=else_result.type),
            )
            yield result_pack
            yield llvm.BrOp(target=merge_label_op, args=result_pack)
        else:
            yield llvm.BrOp(target=merge_label_op, args=_EMPTY_PACK)

        # Merge — block arg receives the result from whichever branch was taken
        yield merge_label_op
        self.current_label = merge_label_op
        self.value_map[op] = merge_result_arg

    def _lower_call(self, op: builtin.CallOp) -> Iterator[dgen.Op]:
        callee_name = op.callee.name
        assert callee_name is not None
        # Unpack PackOp args
        if isinstance(op.args, PackOp):
            mapped_args = [self._map(v) for v in op.args.values]
        else:
            mapped_args = [self._map(op.args)]
        pack = PackOp(values=mapped_args, type=op.args.type)
        yield pack
        llvm_call = llvm.CallOp(
            callee=String().constant(callee_name),
            args=pack,
            type=op.type,
        )
        yield llvm_call
        self.value_map[op] = llvm_call


def lower_builtin_to_llvm(m: Module) -> Module:
    return BuiltinToLLVMLowering().run(m)
