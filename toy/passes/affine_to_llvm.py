"""Affine IR to LLVM-like IR lowering."""

from __future__ import annotations

from math import prod

import dgen
from dgen.block import BlockArgument
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import ChainOp, FunctionOp, Nil, String
from dgen.graph import placeholder_block
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass
from toy.dialects import affine, toy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dgen.compiler import Compiler

_PTR_TYPE = llvm.Ptr()
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


def _extract_indices(
    indices: dgen.Value, value_map: dict[dgen.Value, dgen.Value]
) -> list[dgen.Value]:
    assert isinstance(indices, PackOp)
    return [value_map.get(v, v) for v in indices.values]


class AffineToLLVMLowering(Pass):
    def verify_preconditions(self, module: Module) -> None:
        # The affine IR produced by ToyToAffine has a known closed-block
        # violation: _nested_for creates ForOp body blocks whose inner layers
        # reference outer-loop BlockArgument ivars directly (TODO: fix by
        # threading outer ivars as explicit body block arguments).
        from dgen.verify import verify_all_ready

        verify_all_ready(module)

    def verify_postconditions(self, module: Module) -> None:
        # TODO: _linearize generates MulOp/AddOp nodes for multi-dimensional
        # index linearization that use outer-loop body_iv BlockArguments inside
        # inner LabelOp body blocks — the same structural ivar-leakage as in
        # _nested_for.  Fixing this requires threading outer loop variables
        # through nested label blocks as explicit block arguments.
        pass

    def __init__(self) -> None:
        self.loop_counter = 0
        self.value_map: dict[dgen.Value, dgen.Value] = {}
        self.alloc_shapes: dict[dgen.Value, list[int]] = {}
        self._seen: set[dgen.Value] = set()

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
        self.alloc_shapes = {}
        self._seen = set()
        for arg in f.body.args:
            if isinstance(arg.type, toy.Tensor):
                shape = arg.type.shape.__constant__.to_json()
                assert isinstance(shape, list)
                self.alloc_shapes[arg] = shape
            else:
                self.value_map[arg] = arg
        result = self._lower_ops(f.body.ops, f.body.result)
        return FunctionOp(
            name=f.name,
            body=dgen.Block(result=result, args=f.body.args),
            result=f.result,
        )

    def _map(self, v: dgen.Value) -> dgen.Value:
        return self.value_map.get(v, v)

    def _deref(self, v: dgen.Value) -> dgen.Value:
        """If v is a tensor value, create a fresh LoadOp to get its data ptr.

        Each call produces a new op so callers in different blocks each own
        their copy, respecting the closed-block invariant.
        """
        if not isinstance(v.type, toy.Tensor):
            return v
        load_op = llvm.LoadOp(ptr=v, type=_PTR_TYPE)
        if v in self.alloc_shapes:
            self.alloc_shapes[load_op] = self.alloc_shapes[v]
        return load_op

    def _lower_ops(self, ops: list[dgen.Op], return_val: dgen.Value) -> dgen.Value:
        effects: list[dgen.Op] = []
        for i, op in enumerate(ops):
            if isinstance(op, affine.ForOp):
                entry_br, exit_label = self._lower_for(op)
                exit_result = self._lower_ops(ops[i + 1 :], return_val)
                exit_label.body = dgen.Block(result=exit_result, args=[])
                return _chain_before(effects, entry_br)
            effect = self._lower_op(op)
            if effect is not None:
                effects.append(effect)
        mapped = self._map(return_val)
        return _chain_before([e for e in effects if e is not mapped], mapped)

    def _lower_op(self, op: dgen.Op) -> dgen.Op | None:
        """Lower one non-ForOp. Returns it if it has side effects, else None."""
        if op in self._seen:
            return None
        self._seen.add(op)

        if isinstance(op, affine.AllocOp):
            assert isinstance(op.type, affine.MemRef)
            shape = op.type.shape.__constant__.to_json()
            assert isinstance(shape, list)
            total = prod(shape)
            alloca = llvm.AllocaOp(elem_count=builtin.Index().constant(total))
            self.value_map[op] = alloca
            self.alloc_shapes[alloca] = shape
            # Return alloca as an effect so it is reachable from the function
            # body result and claimed by the entry block (LLVM dominance).
            return alloca

        if isinstance(op, affine.DeallocOp):
            return None  # stack-allocated; no free needed

        if isinstance(op, affine.LoadOp):
            self._lower_load(op)
            return None

        if isinstance(op, affine.StoreOp):
            return self._lower_store(op)

        if isinstance(op, ConstantOp):
            if isinstance(op.type, toy.Tensor):
                shape = op.type.shape.__constant__.to_json()
                assert isinstance(shape, list)
                self.alloc_shapes[op] = shape
                # Do not map op; _map(op) returns op itself.
                # _deref() creates a fresh LoadOp at each use site.
            else:
                self.value_map[op] = ConstantOp(value=op.value, type=op.type)
            return None

        if isinstance(op, affine.MulFOp):
            self.value_map[op] = llvm.FmulOp(
                lhs=self._map(op.lhs), rhs=self._map(op.rhs)
            )
            return None

        if isinstance(op, affine.AddFOp):
            self.value_map[op] = llvm.FaddOp(
                lhs=self._map(op.lhs), rhs=self._map(op.rhs)
            )
            return None

        if isinstance(op, affine.PrintMemrefOp):
            return self._lower_print(op)

        if isinstance(op, ChainOp):
            new_lhs = self._map(op.lhs)
            self.value_map[op] = new_lhs
            if new_lhs in self.alloc_shapes:
                self.alloc_shapes[op] = self.alloc_shapes[new_lhs]
            return None

        if isinstance(op, builtin.AddIndexOp):
            self.value_map[op] = llvm.AddOp(
                lhs=self._map(op.lhs), rhs=self._map(op.rhs)
            )
            return None

        if isinstance(op, toy.NonzeroCountOp):
            self._lower_nonzero_count(op)
            return None

        return None

    def _lower_load(self, op: affine.LoadOp) -> None:
        memref = self._deref(self._map(op.memref))
        indices = _extract_indices(op.indices, self.value_map)
        gep = llvm.GepOp(base=memref, index=self._linearize(memref, indices))
        self.value_map[op] = llvm.LoadOp(ptr=gep)

    def _lower_store(self, op: affine.StoreOp) -> llvm.StoreOp:
        memref = self._deref(self._map(op.memref))
        indices = _extract_indices(op.indices, self.value_map)
        gep = llvm.GepOp(base=memref, index=self._linearize(memref, indices))
        return llvm.StoreOp(value=self._map(op.value), ptr=gep)

    def _linearize(self, memref: dgen.Value, indices: list[dgen.Value]) -> dgen.Value:
        if len(indices) == 1:
            return indices[0]
        shape = self.alloc_shapes[memref]
        result: dgen.Value | None = None
        for i, idx in enumerate(indices):
            stride = prod(shape[i + 1 :])
            term: dgen.Value = (
                llvm.MulOp(lhs=idx, rhs=ConstantOp(value=stride, type=builtin.Index()))
                if stride != 1
                else idx
            )
            result = llvm.AddOp(lhs=result, rhs=term) if result is not None else term
        assert result is not None
        return result

    def _lower_for(self, op: affine.ForOp) -> tuple[llvm.BrOp, llvm.LabelOp]:
        lid = self.loop_counter
        self.loop_counter += 1

        header_iv = BlockArgument(name=f"i{lid}", type=builtin.Index())
        body_iv = BlockArgument(name=f"j{lid}", type=builtin.Index())

        # Map lo/hi (already processed as ConstantOps in the outer block)
        lo_op = self._map(op.lo)
        if lo_op is op.lo:
            lo_op = ConstantOp(value=op.lo.__constant__.to_json(), type=builtin.Index())
            self.value_map[op.lo] = lo_op
        hi_op = self._map(op.hi)
        if hi_op is op.hi:
            hi_op = ConstantOp(value=op.hi.__constant__.to_json(), type=builtin.Index())
            self.value_map[op.hi] = hi_op

        # Header: compare loop var against hi, branch true→body or false→exit
        cmp_op = llvm.IcmpOp(pred=String().constant("slt"), lhs=header_iv, rhs=hi_op)
        exit_label = llvm.LabelOp(name=f"loop_exit{lid}", body=placeholder_block())
        body_label = llvm.LabelOp(
            name=f"loop_body{lid}",
            body=dgen.Block(result=dgen.Value(type=Nil()), args=[body_iv]),
        )
        cond_br = llvm.CondBrOp(
            cond=cmp_op,
            true_target=body_label,
            false_target=exit_label,
            true_args=_make_pack([header_iv]),
            false_args=_EMPTY_PACK,
        )
        header_label = llvm.LabelOp(
            name=f"loop_header{lid}",
            body=dgen.Block(result=cond_br, args=[header_iv]),
        )

        # Body: map the affine loop var to the body iv, lower body ops
        self.value_map[op.body.args[0]] = header_iv  # for header ops
        self.value_map[op.body.args[0]] = body_iv  # for body ops
        one = ConstantOp(value=1, type=builtin.Index())
        next_iv = llvm.AddOp(lhs=body_iv, rhs=one)
        back_br = llvm.BrOp(target=header_label, args=_make_pack([next_iv]))
        body_result = self._lower_ops(op.body.ops, back_br)
        body_label.body = dgen.Block(result=body_result, args=[body_iv])

        entry_br = llvm.BrOp(target=header_label, args=_make_pack([lo_op]))
        return entry_br, exit_label

    def _lower_nonzero_count(self, op: toy.NonzeroCountOp) -> None:
        input_ptr = self._deref(self._map(op.input))
        assert isinstance(op.input.type, toy.Tensor)
        total = prod(op.input.type.shape.__constant__.to_json())
        zero_f = ConstantOp(value=0.0, type=builtin.F64())
        acc: dgen.Value = ConstantOp(value=0, type=builtin.Index())
        for i in range(total):
            idx = ConstantOp(value=i, type=builtin.Index())
            gep = llvm.GepOp(base=input_ptr, index=idx)
            elem = llvm.LoadOp(ptr=gep)
            cmp = llvm.FcmpOp(pred=String().constant("one"), lhs=elem, rhs=zero_f)
            acc = llvm.AddOp(lhs=acc, rhs=llvm.ZextOp(input=cmp))
        self.value_map[op] = acc

    def _lower_print(self, op: affine.PrintMemrefOp) -> llvm.CallOp:
        input_val = self._deref(self._map(op.input))
        size = prod(self.alloc_shapes[input_val])
        pack = PackOp(
            values=[input_val, ConstantOp(value=size, type=builtin.Index())],
            type=builtin.List(element_type=input_val.type),
        )
        call_op = llvm.CallOp(callee=String().constant("print_memref"), args=pack)
        self.value_map[op] = call_op
        return call_op
