"""Lower builtin dialect ops to LLVM dialect ops.

Handles: AddIndexOp, SubtractIndexOp, EqualIndexOp, IfOp, CallOp.
Passes through unchanged: ConstantOp, PackOp, ReturnOp, and any LLVM dialect ops.
"""

from __future__ import annotations

from collections.abc import Iterator

import dgen
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import FunctionOp, Nil, String
from dgen.graph import chain_body, group_into_blocks, placeholder_block
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass


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

        # Phase 1: Generate ops linearly (labels as boundary markers)
        # Flatten existing label bodies so their ops are processed too.
        flat_ops: list[dgen.Op] = []
        for op in f.body.ops:
            if isinstance(op, llvm.LabelOp) and op.body.ops:
                flat_ops.append(op)
                for body_op in op.body.ops:
                    flat_ops.extend(self.lower_op(body_op))
            else:
                flat_ops.extend(self.lower_op(op))

        # Phase 2: Group into basic blocks and build label body blocks
        entry_ops, label_groups = group_into_blocks(flat_ops)
        for label_op, body_ops in label_groups:
            label_op.body = dgen.Block(result=chain_body(body_ops), ops=body_ops)

        func_ops: list[dgen.Op] = entry_ops + [lg[0] for lg in label_groups]
        return FunctionOp(
            name=f.name,
            body=dgen.Block(ops=func_ops, args=f.body.args),
            result=f.result,
        )

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
        elif isinstance(op, builtin.ReturnOp):
            if isinstance(op.value, Nil):
                yield op
            else:
                mapped = self._map(op.value)
                if mapped is op.value:
                    yield op
                else:
                    yield builtin.ReturnOp(value=mapped, type=op.type)
        else:
            # Pass through ConstantOp, PackOp, LLVM ops, etc. unchanged
            yield op

    def _lower_if(self, op: builtin.IfOp) -> Iterator[dgen.Op]:
        if_id = self.if_counter
        self.if_counter += 1

        then_label_op = llvm.LabelOp(name=f"then_{if_id}", body=placeholder_block())
        else_label_op = llvm.LabelOp(name=f"else_{if_id}", body=placeholder_block())
        merge_label_op = llvm.LabelOp(name=f"merge_{if_id}", body=placeholder_block())

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
        )

        # Then block
        yield then_label_op
        self.current_label = then_label_op
        then_result: dgen.Value | None = None
        for child in op.then_body.ops:
            if isinstance(child, builtin.ReturnOp) and not isinstance(child.value, Nil):
                then_result = self._map(child.value)
                yield llvm.BrOp(target=merge_label_op)
            else:
                yield from self.lower_op(child)
        then_source_label = self.current_label

        # Else block
        yield else_label_op
        self.current_label = else_label_op
        else_result: dgen.Value | None = None
        for child in op.else_body.ops:
            if isinstance(child, builtin.ReturnOp) and not isinstance(child.value, Nil):
                else_result = self._map(child.value)
                yield llvm.BrOp(target=merge_label_op)
            else:
                yield from self.lower_op(child)
        else_source_label = self.current_label

        # Merge with phi
        yield merge_label_op
        self.current_label = merge_label_op
        assert then_result is not None and else_result is not None
        phi_op = llvm.PhiOp(
            a=then_result,
            b=else_result,
            label_a=then_source_label,
            label_b=else_source_label,
        )
        yield phi_op
        self.value_map[op] = phi_op

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
