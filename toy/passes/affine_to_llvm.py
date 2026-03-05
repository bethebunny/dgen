"""Ch6: Affine IR to LLVM-like IR lowering."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from math import prod

import dgen
from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import FunctionOp, Nil, PackOp, String
from dgen.layout import Array
from dgen.module import ConstantOp, Function, Module
from toy.dialects import affine, toy


def _extract_list_elements(
    list_val: dgen.Value,
    value_map: dict[dgen.Value, dgen.Value],
) -> list[dgen.Value]:
    """Extract elements from a PackOp, mapping values through value_map."""
    assert isinstance(list_val, PackOp)
    return [value_map.get(v, v) for v in list_val.values]


class AffineToLLVMLowering:
    def __init__(self) -> None:
        self.loop_counter = 0
        self.value_map: dict[dgen.Value, dgen.Value] = {}  # affine -> llvm
        self.alloc_shapes: dict[dgen.Value, list[int]] = {}  # llvm alloca -> shape
        self.alloc_sizes: dict[dgen.Value, int] = {}  # llvm alloca -> total size
        self.current_label = "entry"

    def lower_module(self, m: Module) -> Module:
        functions = [self.lower_function(f) for f in m.functions]
        return Module(functions=functions)

    def lower_function(self, f: FunctionOp) -> FunctionOp:
        self.loop_counter = 0
        self.value_map = {}
        self.alloc_shapes = {}
        self.alloc_sizes = {}
        self.current_label = "entry"
        # Register block args (function parameters)
        for arg in f.body.args:
            self.value_map[arg] = arg
            if isinstance(arg.type, toy.TensorType):
                shape = arg.type.unpack_shape()
                self.alloc_shapes[arg] = shape
                self.alloc_sizes[arg] = prod(shape)
        ops = []
        for op in f.body.ops:
            ops.extend(self.lower_op(op))
        return FunctionOp(
            name=f.name,
            body=dgen.Block(ops=ops, args=f.body.args),
            type=Function(result=f.type.result),
        )

    def _map(self, old: dgen.Value) -> dgen.Value:
        """Resolve an affine Value to its LLVM counterpart."""
        return self.value_map.get(old, old)

    def lower_op(self, op: dgen.Op) -> Iterator[dgen.Op]:
        if isinstance(op, affine.AllocOp):
            yield from self._lower_alloc(op)
        elif isinstance(op, affine.DeallocOp):
            pass  # Stack alloc, no free needed
        elif isinstance(op, affine.LoadOp):
            yield from self._lower_load(op)
        elif isinstance(op, affine.StoreOp):
            yield from self._lower_store(op)
        elif isinstance(op, affine.ForOp):
            yield from self._lower_for(op)
        elif isinstance(op, ConstantOp):
            new_op = ConstantOp(value=op.value, type=op.type)
            yield new_op
            self.value_map[op] = new_op
            if isinstance(op.value.layout, Array) and isinstance(
                op.type, toy.TensorType
            ):
                shape = op.type.unpack_shape()
                self.alloc_shapes[new_op] = shape
                self.alloc_sizes[new_op] = prod(shape)
        elif isinstance(op, affine.MulFOp):
            llvm_op = llvm.FMulOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            yield llvm_op
            self.value_map[op] = llvm_op
        elif isinstance(op, affine.AddFOp):
            llvm_op = llvm.FAddOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            yield llvm_op
            self.value_map[op] = llvm_op
        elif isinstance(op, affine.PrintMemrefOp):
            yield from self._lower_print(op)
        elif isinstance(op, builtin.AddIndexOp):
            llvm_op = llvm.AddOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            yield llvm_op
            self.value_map[op] = llvm_op
        elif isinstance(op, toy.NonzeroCountOp):
            yield from self._lower_nonzero_count(op)
        elif isinstance(op, builtin.ReturnOp):
            if isinstance(op.value, Nil):
                yield builtin.ReturnOp()
            else:
                yield builtin.ReturnOp(value=self._map(op.value))

    def _lower_alloc(self, op: affine.AllocOp) -> Iterator[dgen.Op]:
        assert isinstance(op.type, affine.MemRefType)
        shape = op.type.shape.__constant__.to_json()  # MemRefType, not TensorType
        total = prod(shape)
        alloca_op = llvm.AllocaOp(elem_count=builtin.IndexType().constant(total))
        yield alloca_op
        self.value_map[op] = alloca_op
        self.alloc_shapes[alloca_op] = shape
        self.alloc_sizes[alloca_op] = total

    def _lower_load(self, op: affine.LoadOp) -> Iterator[dgen.Op]:
        memref_val = self._map(op.memref)
        index_vals = _extract_list_elements(op.indices, self.value_map)
        linear = yield from self._linearize_indices(memref_val, index_vals)
        ptr_op = llvm.GepOp(base=memref_val, index=linear)
        yield ptr_op
        load_op = llvm.LoadOp(ptr=ptr_op)
        yield load_op
        self.value_map[op] = load_op

    def _lower_store(self, op: affine.StoreOp) -> Iterator[dgen.Op]:
        memref_val = self._map(op.memref)
        index_vals = _extract_list_elements(op.indices, self.value_map)
        linear = yield from self._linearize_indices(memref_val, index_vals)
        ptr_op = llvm.GepOp(base=memref_val, index=linear)
        yield ptr_op
        yield llvm.StoreOp(value=self._map(op.value), ptr=ptr_op)

    def _linearize_indices(
        self, memref: dgen.Value, indices: list[dgen.Value]
    ) -> Generator[dgen.Op, None, dgen.Value]:
        if len(indices) == 1:
            return indices[0]

        shape = self.alloc_shapes[memref]

        result_val = None
        for i, idx_val in enumerate(indices):
            stride = 1
            for j in range(i + 1, len(shape)):
                stride *= shape[j]

            if stride == 1:
                if result_val is None:
                    result_val = idx_val
                else:
                    add_op = llvm.AddOp(lhs=result_val, rhs=idx_val)
                    yield add_op
                    result_val = add_op
            else:
                stride_op = ConstantOp(value=stride, type=builtin.IndexType())
                yield stride_op
                mul_op = llvm.MulOp(lhs=idx_val, rhs=stride_op)
                yield mul_op
                if result_val is None:
                    result_val = mul_op
                else:
                    add_op = llvm.AddOp(lhs=result_val, rhs=mul_op)
                    yield add_op
                    result_val = add_op

        assert result_val is not None
        return result_val

    def _lower_for(self, op: affine.ForOp) -> Iterator[dgen.Op]:
        loop_id = self.loop_counter
        self.loop_counter += 1

        header_label = f"loop_header{loop_id}"
        body_label = f"loop_body{loop_id}"
        exit_label = f"loop_exit{loop_id}"

        # Init: constant lo, br header
        init_op = ConstantOp(
            value=op.lo.__constant__.to_json(), type=builtin.IndexType()
        )
        yield init_op
        prev_label = self.current_label
        yield llvm.BrOp(dest=String().constant(header_label))

        # Header label
        yield llvm.LabelOp(label_name=String().constant(header_label))
        self.current_label = header_label

        # Phi node for loop variable (back-edge value patched after body)
        back_edge = dgen.Value(type=builtin.IndexType())  # placeholder, patched below
        phi_op = llvm.PhiOp(
            values=[init_op, back_edge],
            labels=[String().constant(prev_label), String().constant(body_label)],
        )
        yield phi_op
        # Map the affine loop variable to the LLVM phi node
        self.value_map[op.body.args[0]] = phi_op

        # Compare and branch
        hi_op = ConstantOp(value=op.hi.__constant__.to_json(), type=builtin.IndexType())
        yield hi_op
        cmp_op = llvm.IcmpOp(pred=String().constant("slt"), lhs=phi_op, rhs=hi_op)
        yield cmp_op
        yield llvm.CondBrOp(
            cond=cmp_op,
            true_dest=String().constant(body_label),
            false_dest=String().constant(exit_label),
        )

        # Body label
        yield llvm.LabelOp(label_name=String().constant(body_label))
        self.current_label = body_label

        # Lower body ops
        for child_op in op.body.ops:
            yield from self.lower_op(child_op)

        # Patch phi back-edge label to actual current block
        phi_op.labels[1] = String().constant(self.current_label)

        # Increment and branch back
        one_op = ConstantOp(value=1, type=builtin.IndexType())
        yield one_op
        next_op = llvm.AddOp(lhs=phi_op, rhs=one_op)
        yield next_op
        # Patch phi back-edge value
        phi_op.values[1] = next_op
        yield llvm.BrOp(dest=String().constant(header_label))

        # Exit label
        yield llvm.LabelOp(label_name=String().constant(exit_label))
        self.current_label = exit_label

    def _lower_nonzero_count(self, op: toy.NonzeroCountOp) -> Iterator[dgen.Op]:
        """Unrolled nonzero_count: count non-zero elements in a tensor."""
        input_val = self._map(op.input)
        assert isinstance(op.input.type, toy.TensorType)
        total = prod(op.input.type.unpack_shape())

        zero_f = ConstantOp(value=0.0, type=builtin.F64Type())
        yield zero_f
        acc = ConstantOp(value=0, type=builtin.IndexType())
        yield acc

        for i in range(total):
            idx = ConstantOp(value=i, type=builtin.IndexType())
            yield idx
            gep = llvm.GepOp(base=input_val, index=idx)
            yield gep
            elem = llvm.LoadOp(ptr=gep)
            yield elem
            cmp = llvm.FcmpOp(pred=String().constant("one"), lhs=elem, rhs=zero_f)
            yield cmp
            ext = llvm.ZextOp(input=cmp)
            yield ext
            new_acc = llvm.AddOp(lhs=acc, rhs=ext)
            yield new_acc
            acc = new_acc

        self.value_map[op] = acc

    def _lower_print(self, op: affine.PrintMemrefOp) -> Iterator[dgen.Op]:
        input_val = self._map(op.input)
        size = self.alloc_sizes[input_val]
        size_op = ConstantOp(value=size, type=builtin.IndexType())
        yield size_op
        yield llvm.CallOp(
            callee=String().constant("print_memref"),
            args=[input_val, size_op],
        )


def lower_to_llvm(m: Module) -> Module:
    lowering = AffineToLLVMLowering()
    return lowering.lower_module(m)
