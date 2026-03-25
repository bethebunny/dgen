"""Structured IR to LLVM-like IR lowering.

Lowers memory, control_flow, and algebra dialect ops to goto + llvm dialect ops.
"""

from __future__ import annotations

from collections.abc import Callable
from math import prod

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra, builtin, control_flow, goto, index, llvm
from dgen.dialects.builtin import ChainOp, F64, Nil, String
from dgen.dialects.function import Function, FunctionOp
from dgen.graph import placeholder_block
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass
from toy.dialects import memory, toy

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


class StructuredToLLVM(Pass):
    def __init__(self) -> None:
        self.loop_counter = 0
        self.value_map: dict[dgen.Value, dgen.Value] = {}
        self.alloc_shapes: dict[dgen.Value, list[int]] = {}
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
        self.alloc_shapes = {}
        self._seen = set()
        for arg in f.body.args:
            if isinstance(arg.type, toy.Tensor):
                shape = arg.type.shape.__constant__.to_json()
                assert isinstance(shape, list)
                self.alloc_shapes[arg] = shape
                self.value_map[arg] = arg  # allow _lower_for to thread tensor args
            else:
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
                # Exit captures: outer affine ivars + enclosing header_selfs.
                exit_captures: list[dgen.Value] = [
                    *self._header_selfs,
                    *(self._map(a) for a in op.body.args[1:]),
                ]
                exit_result = self._lower_ops(ops[i + 1 :], return_builder)
                exit_label.body = dgen.Block(result=exit_result, captures=exit_captures)
                # Clean up: remove this loop's header_self.
                self._header_selfs.pop()
                del self.value_map[op]
                return _chain_before(effects, entry_br)
            effect = self._lower_op(op)
            if effect is not None:
                effects.append(effect)
        mapped = return_builder()
        return _chain_before([e for e in effects if e is not mapped], mapped)

    def _lower_op(self, op: dgen.Op) -> dgen.Op | None:
        """Lower one non-ForOp. Returns it if it has side effects, else None."""
        if op in self._seen:
            return None
        self._seen.add(op)

        if isinstance(op, memory.AllocOp):
            assert isinstance(op.type, memory.MemRef)
            shape = op.type.shape.__constant__.to_json()
            assert isinstance(shape, list)
            total = prod(shape)
            # Heap-allocate: malloc(total * 8) for 8-byte doubles.
            # Stack alloca would be invalid if the pointer escapes the function.
            byte_count = ConstantOp(value=total * 8, type=index.Index())
            malloc_args = PackOp(
                values=[byte_count], type=builtin.List(element_type=index.Index())
            )
            malloc_op = llvm.CallOp(
                callee=String().constant("malloc"), args=malloc_args, type=_PTR_TYPE
            )
            self.value_map[op] = malloc_op
            self.alloc_shapes[malloc_op] = shape
            # Return malloc as an effect so it is reachable from the function
            # body result and claimed by the entry block (LLVM dominance).
            return malloc_op
        if isinstance(op, memory.DeallocOp):
            return None  # heap-allocated; no free (leak for now)
        if isinstance(op, memory.LoadOp):
            self._lower_load(op)
            return None

        if isinstance(op, memory.StoreOp):
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

        if isinstance(op, algebra.MultiplyOp):
            left = self._map(op.left)
            right = self._map(op.right)
            if isinstance(op.type, F64):
                self.value_map[op] = llvm.FmulOp(lhs=left, rhs=right)
            else:
                self.value_map[op] = llvm.MulOp(lhs=left, rhs=right)
            return None

        if isinstance(op, algebra.AddOp):
            left = self._map(op.left)
            right = self._map(op.right)
            if isinstance(op.type, F64):
                self.value_map[op] = llvm.FaddOp(lhs=left, rhs=right)
            else:
                self.value_map[op] = llvm.AddOp(lhs=left, rhs=right)
            return None

        if isinstance(op, memory.PrintMemrefOp):
            return self._lower_print(op)

        if isinstance(op, ChainOp):
            new_lhs = self._map(op.lhs)
            self.value_map[op] = new_lhs
            if new_lhs in self.alloc_shapes:
                self.alloc_shapes[op] = self.alloc_shapes[new_lhs]
            return None

        if isinstance(op, toy.NonzeroCountOp):
            self._lower_nonzero_count(op)
            return None

        return None

    def _lower_load(self, op: memory.LoadOp) -> None:
        memref = self._deref(self._map(op.memref))
        indices = _extract_indices(op.indices, self.value_map)
        gep = llvm.GepOp(base=memref, index=self._linearize(memref, indices))
        self.value_map[op] = llvm.LoadOp(ptr=gep)

    def _lower_store(self, op: memory.StoreOp) -> llvm.StoreOp:
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
                llvm.MulOp(lhs=idx, rhs=ConstantOp(value=stride, type=index.Index()))
                if stride != 1
                else idx
            )
            result = llvm.AddOp(lhs=result, rhs=term) if result is not None else term
        assert result is not None
        return result

    def _lower_for(self, op: control_flow.ForOp) -> tuple[goto.BranchOp, goto.LabelOp]:
        """Lower one ForOp to LLVM header/body/exit labels.

        The header gets a %self block parameter. Body and exit blocks capture
        outer-scope values directly (no threading through args). Only the loop
        induction variable is a block arg (it varies per iteration/predecessor).

        Returns (entry_br, exit_label) where exit_label.body is a placeholder
        to be filled by the caller with the post-loop continuation.
        """
        lid = self.loop_counter
        self.loop_counter += 1

        header_iv = BlockArgument(name=f"i{lid}", type=index.Index())
        body_iv = BlockArgument(name=f"j{lid}", type=index.Index())
        header_self = BlockArgument(name="self", type=goto.Label())

        # Map lower_bound/upper_bound
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

        # Header: compare iv against hi, branch true→body or false→exit.
        cmp_op = llvm.IcmpOp(pred=String().constant("slt"), lhs=header_iv, rhs=hi_op)
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
            body=dgen.Block(
                result=cond_br,
                parameters=[header_self],
                args=[header_iv],
            ),
        )

        # Register header_self so nested blocks can capture it.
        self.value_map[op] = header_self
        self._header_selfs.append(header_self)

        # Body captures: header_self + enclosing header_selfs + outer affine ivars.
        body_captures: list[dgen.Value] = [
            *self._header_selfs,
            *(self._map(a) for a in op.body.args[1:]),
        ]

        # Map affine loop var for the body.
        outer_affine_ivs = list(op.body.args[1:])
        saved_mappings = {
            a: self.value_map[a] for a in outer_affine_ivs if a in self.value_map
        }
        self.value_map[op.body.args[0]] = body_iv

        def _make_back_br() -> dgen.Value:
            one = ConstantOp(value=1, type=index.Index())
            next_iv = llvm.AddOp(lhs=self.value_map[op.body.args[0]], rhs=one)
            return goto.BranchOp(target=self._map(op), arguments=_make_pack([next_iv]))

        body_result = self._lower_ops(op.body.ops, _make_back_br)
        body_label.body = dgen.Block(
            result=body_result, args=[body_iv], captures=body_captures
        )

        # Restore outer iv mappings.
        for a, v in saved_mappings.items():
            self.value_map[a] = v

        # Header captures: enclosing header_selfs (not our own) + outer affine ivars.
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

        entry_br = goto.BranchOp(
            target=header_label,
            arguments=_make_pack([lo_op]),
        )
        return entry_br, exit_label

    def _lower_nonzero_count(self, op: toy.NonzeroCountOp) -> None:
        input_ptr = self._deref(self._map(op.input))
        assert isinstance(op.input.type, toy.Tensor)
        total = prod(op.input.type.shape.__constant__.to_json())
        zero_f = ConstantOp(value=0.0, type=builtin.F64())
        acc: dgen.Value = ConstantOp(value=0, type=index.Index())
        for i in range(total):
            idx = ConstantOp(value=i, type=index.Index())
            gep = llvm.GepOp(base=input_ptr, index=idx)
            elem = llvm.LoadOp(ptr=gep)
            cmp = llvm.FcmpOp(pred=String().constant("one"), lhs=elem, rhs=zero_f)
            acc = llvm.AddOp(lhs=acc, rhs=llvm.ZextOp(input=cmp))
        self.value_map[op] = acc

    def _lower_print(self, op: memory.PrintMemrefOp) -> llvm.CallOp:
        input_val = self._deref(self._map(op.input))
        size = prod(self.alloc_shapes[input_val])
        pack = PackOp(
            values=[input_val, ConstantOp(value=size, type=index.Index())],
            type=builtin.List(element_type=input_val.type),
        )
        call_op = llvm.CallOp(callee=String().constant("print_memref"), args=pack)
        self.value_map[op] = call_op
        return call_op
