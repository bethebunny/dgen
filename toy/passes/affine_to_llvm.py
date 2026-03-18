"""Ch6: Affine IR to LLVM-like IR lowering."""

from __future__ import annotations

from math import prod

import dgen
from dgen.block import BlockArgument
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import ChainOp, FunctionOp, Nil, String
from dgen.graph import chain_body, placeholder_block
from dgen.module import ConstantOp, Module, PackOp
from dgen.passes.pass_ import Pass
from toy.dialects import affine, toy

_PTR_TYPE = llvm.Ptr()
_EMPTY_PACK = PackOp(values=[], type=builtin.List(element_type=Nil()))


def _extract_list_elements(
    list_val: dgen.Value,
    value_map: dict[dgen.Value, dgen.Value],
) -> list[dgen.Value]:
    """Extract elements from a PackOp, mapping values through value_map."""
    assert isinstance(list_val, PackOp)
    return [value_map.get(v, v) for v in list_val.values]


def _make_pack(values: list[dgen.Value]) -> PackOp:
    """Create a PackOp wrapping the given values."""
    if not values:
        return _EMPTY_PACK
    return PackOp(values=values, type=builtin.List(element_type=values[0].type))


def _chain_before(effects: list[dgen.Op], terminal: dgen.Value) -> dgen.Value:
    """Chain effects before terminal so they execute in list order."""
    result: dgen.Value = terminal
    for effect in reversed(effects):
        result = ChainOp(lhs=effect, rhs=result, type=terminal.type)
    return result


class AffineToLLVMLowering(Pass):
    def __init__(self) -> None:
        self.loop_counter = 0
        self.value_map: dict[dgen.Value, dgen.Value] = {}  # affine -> llvm
        self.alloc_shapes: dict[dgen.Value, list[int]] = {}  # llvm alloca -> shape
        self.alloc_sizes: dict[dgen.Value, int] = {}  # llvm alloca -> total size
        self._seen: set[dgen.Value] = set()  # ops already processed

    def run(self, m: Module) -> Module:
        functions = [self._lower_function(f) for f in m.functions]
        return Module(functions=functions)

    def _lower_function(self, f: FunctionOp) -> FunctionOp:
        self.loop_counter = 0
        self.value_map = {}
        self.alloc_shapes = {}
        self.alloc_sizes = {}
        self._seen = set()
        # Register block args (function parameters)
        for arg in f.body.args:
            if isinstance(arg.type, toy.Tensor):
                # arg is a Span struct ptr; load the data ptr from it
                load_op = llvm.LoadOp(ptr=arg, type=_PTR_TYPE)
                self.value_map[arg] = load_op
                shape = arg.type.shape.__constant__.to_json()
                self.alloc_shapes[load_op] = shape
                self.alloc_sizes[load_op] = prod(shape)
            else:
                self.value_map[arg] = arg

        result = self._lower_ops(f.body.ops, f.body.result)
        return FunctionOp(
            name=f.name,
            body=dgen.Block(result=result, args=f.body.args),
            result=f.result,
        )

    def _lower_ops(self, ops: list[dgen.Op], return_val: dgen.Value) -> dgen.Value:
        """Lower a sequence of ops, returning the block result with effects chained."""
        effects: list[dgen.Op] = []
        for i, op in enumerate(ops):
            if isinstance(op, affine.ForOp):
                entry_br, exit_label = self._lower_for(op)
                exit_result = self._lower_ops(ops[i + 1 :], return_val)
                exit_label.body = dgen.Block(
                    result=exit_result, args=exit_label.body.args
                )
                return _chain_before(effects, entry_br)
            effect = self._lower_single_op(op)
            if effect is not None:
                effects.append(effect)
        mapped = self.value_map.get(return_val, return_val)
        return _chain_before([e for e in effects if e is not mapped], mapped)

    def _map(self, old: dgen.Value) -> dgen.Value:
        """Resolve an affine Value to its LLVM counterpart."""
        return self.value_map.get(old, old)

    def _lower_single_op(self, op: dgen.Op) -> dgen.Op | None:
        """Lower one non-ForOp. Returns the op if it's a side effect, else None."""
        if op in self._seen:
            return None
        self._seen.add(op)
        if isinstance(op, affine.AllocOp):
            assert isinstance(op.type, affine.MemRef)
            shape = op.type.shape.__constant__.to_json()
            total = prod(shape)
            alloca_op = llvm.AllocaOp(elem_count=builtin.Index().constant(total))
            self.value_map[op] = alloca_op
            self.alloc_shapes[alloca_op] = shape
            self.alloc_sizes[alloca_op] = total
            return None
        if isinstance(op, affine.DeallocOp):
            return None  # Stack alloc, no free needed
        if isinstance(op, affine.LoadOp):
            self._lower_load(op)
            return None
        if isinstance(op, affine.StoreOp):
            return self._lower_store(op)
        if isinstance(op, ConstantOp):
            new_op = ConstantOp(value=op.value, type=op.type)
            if isinstance(op.type, toy.Tensor):
                # new_op is a Span struct ptr; load the data ptr from it
                load_op = llvm.LoadOp(ptr=new_op, type=_PTR_TYPE)
                self.value_map[op] = load_op
                shape = op.type.shape.__constant__.to_json()
                self.alloc_shapes[load_op] = shape
                self.alloc_sizes[load_op] = prod(shape)
            else:
                self.value_map[op] = new_op
            return None
        if isinstance(op, affine.MulFOp):
            llvm_op = llvm.FmulOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            self.value_map[op] = llvm_op
            return None
        if isinstance(op, affine.AddFOp):
            llvm_op = llvm.FaddOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            self.value_map[op] = llvm_op
            return None
        if isinstance(op, affine.PrintMemrefOp):
            return self._lower_print(op)
        if isinstance(op, ChainOp):
            # ChainOp is a transparent alias for its lhs; propagate shape metadata.
            new_lhs = self._map(op.lhs)
            self.value_map[op] = new_lhs
            if new_lhs in self.alloc_shapes:
                self.alloc_shapes[op] = self.alloc_shapes[new_lhs]
                self.alloc_sizes[op] = self.alloc_sizes[new_lhs]
            return None
        if isinstance(op, builtin.AddIndexOp):
            llvm_op = llvm.AddOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            self.value_map[op] = llvm_op
            return None
        if isinstance(op, toy.NonzeroCountOp):
            self._lower_nonzero_count(op)
            return None
        return None

    def _lower_load(self, op: affine.LoadOp) -> None:
        memref_val = self._map(op.memref)
        index_vals = _extract_list_elements(op.indices, self.value_map)
        linear = self._linearize_indices(memref_val, index_vals)
        ptr_op = llvm.GepOp(base=memref_val, index=linear)
        load_op = llvm.LoadOp(ptr=ptr_op)
        self.value_map[op] = load_op

    def _lower_store(self, op: affine.StoreOp) -> llvm.StoreOp:
        memref_val = self._map(op.memref)
        index_vals = _extract_list_elements(op.indices, self.value_map)
        linear = self._linearize_indices(memref_val, index_vals)
        ptr_op = llvm.GepOp(base=memref_val, index=linear)
        return llvm.StoreOp(value=self._map(op.value), ptr=ptr_op)

    def _linearize_indices(
        self, memref: dgen.Value, indices: list[dgen.Value]
    ) -> dgen.Value:
        if len(indices) == 1:
            return indices[0]

        shape = self.alloc_shapes[memref]
        result_val: dgen.Value | None = None
        for i, idx_val in enumerate(indices):
            stride = prod(shape[i + 1 :])
            if stride == 1:
                if result_val is None:
                    result_val = idx_val
                else:
                    add_op = llvm.AddOp(lhs=result_val, rhs=idx_val)
                    result_val = add_op
            else:
                stride_op = ConstantOp(value=stride, type=builtin.Index())
                mul_op = llvm.MulOp(lhs=idx_val, rhs=stride_op)
                if result_val is None:
                    result_val = mul_op
                else:
                    add_op = llvm.AddOp(lhs=result_val, rhs=mul_op)
                    result_val = add_op

        assert result_val is not None
        return result_val

    def _lower_for(self, op: affine.ForOp) -> tuple[llvm.BrOp, llvm.LabelOp]:
        loop_id = self.loop_counter
        self.loop_counter += 1

        header_label = f"loop_header{loop_id}"
        body_label = f"loop_body{loop_id}"
        exit_label = f"loop_exit{loop_id}"

        # lo may already be mapped (shared across ForOps or processed earlier).
        init_op = self._map(op.lo)
        if init_op is op.lo:
            init_op = ConstantOp(
                value=op.lo.__constant__.to_json(), type=builtin.Index()
            )
            self.value_map[op.lo] = init_op

        # Collect alloca pointers to thread through the loop.
        alloca_entries: list[dgen.Value] = [
            v for v in self.alloc_shapes if not isinstance(v, ChainOp)
        ]

        # --- Block args for header: loop_var + alloca ptrs ---
        header_loop_var = BlockArgument(name=f"i{loop_id}", type=builtin.Index())
        header_alloca_args = [BlockArgument(type=_PTR_TYPE) for _ in alloca_entries]
        header_args = [header_loop_var] + header_alloca_args

        # --- Block args for body: loop_var + alloca ptrs ---
        body_loop_var = BlockArgument(name=f"j{loop_id}", type=builtin.Index())
        body_alloca_args = [BlockArgument(type=_PTR_TYPE) for _ in alloca_entries]
        body_args = [body_loop_var] + body_alloca_args

        # --- Build header block ---
        self.value_map[op.body.args[0]] = header_loop_var
        hi_op = self._map(op.hi)
        if hi_op is op.hi:
            hi_op = ConstantOp(value=op.hi.__constant__.to_json(), type=builtin.Index())
            self.value_map[op.hi] = hi_op

        cmp_op = llvm.IcmpOp(
            pred=String().constant("slt"), lhs=header_loop_var, rhs=hi_op
        )

        exit_label_op = llvm.LabelOp(name=exit_label, body=placeholder_block())
        body_label_op = llvm.LabelOp(
            name=body_label,
            body=dgen.Block(result=dgen.Value(type=Nil()), args=body_args),
        )

        cond_true_pack = _make_pack([header_loop_var] + header_alloca_args)
        cond_br = llvm.CondBrOp(
            cond=cmp_op,
            true_target=body_label_op,
            false_target=exit_label_op,
            true_args=cond_true_pack,
            false_args=_EMPTY_PACK,
        )
        header_label_op = llvm.LabelOp(
            name=header_label,
            body=dgen.Block(result=chain_body([cmp_op, cond_br]), args=header_args),
        )

        # --- Build body block ---
        saved_value_map = dict(self.value_map)
        saved_alloc_shapes = dict(self.alloc_shapes)
        saved_alloc_sizes = dict(self.alloc_sizes)

        self.value_map[op.body.args[0]] = body_loop_var
        for entry_alloca, body_arg in zip(alloca_entries, body_alloca_args):
            self.alloc_shapes[body_arg] = self.alloc_shapes[entry_alloca]
            self.alloc_sizes[body_arg] = self.alloc_sizes[entry_alloca]
            del self.alloc_shapes[entry_alloca]
            del self.alloc_sizes[entry_alloca]
            for key, val in list(self.value_map.items()):
                if val is entry_alloca:
                    self.value_map[key] = body_arg

        # Build back-edge ops (pure; reachable from back_br via its args).
        one_op = ConstantOp(value=1, type=builtin.Index())
        next_op = llvm.AddOp(lhs=body_loop_var, rhs=one_op)
        back_pack = _make_pack([next_op] + body_alloca_args)
        back_br = llvm.BrOp(target=header_label_op, args=back_pack)

        body_result = self._lower_ops(op.body.ops, back_br)
        body_label_op.body = dgen.Block(result=body_result, args=body_args)

        self.value_map = saved_value_map
        self.alloc_shapes = saved_alloc_shapes
        self.alloc_sizes = saved_alloc_sizes

        entry_pack = _make_pack([init_op] + alloca_entries)
        entry_br = llvm.BrOp(target=header_label_op, args=entry_pack)
        return entry_br, exit_label_op

    def _lower_nonzero_count(self, op: toy.NonzeroCountOp) -> None:
        """Unrolled nonzero_count: count non-zero elements in a tensor."""
        input_val = self._map(op.input)
        assert isinstance(op.input.type, toy.Tensor)
        total = prod(op.input.type.shape.__constant__.to_json())

        zero_f = ConstantOp(value=0.0, type=builtin.F64())
        acc: dgen.Value = ConstantOp(value=0, type=builtin.Index())

        for i in range(total):
            idx = ConstantOp(value=i, type=builtin.Index())
            gep = llvm.GepOp(base=input_val, index=idx)
            elem = llvm.LoadOp(ptr=gep)
            cmp = llvm.FcmpOp(pred=String().constant("one"), lhs=elem, rhs=zero_f)
            ext = llvm.ZextOp(input=cmp)
            acc = llvm.AddOp(lhs=acc, rhs=ext)

        self.value_map[op] = acc

    def _lower_print(self, op: affine.PrintMemrefOp) -> llvm.CallOp:
        input_val = self._map(op.input)
        size = self.alloc_sizes[input_val]
        size_op = ConstantOp(value=size, type=builtin.Index())
        pack = PackOp(
            values=[input_val, size_op], type=builtin.List(element_type=input_val.type)
        )
        call_op = llvm.CallOp(
            callee=String().constant("print_memref"),
            args=pack,
        )
        self.value_map[op] = call_op
        return call_op


def lower_to_llvm(m: Module) -> Module:
    return AffineToLLVMLowering().run(m)
