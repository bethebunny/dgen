"""Ch6: Affine IR to LLVM-like IR lowering."""

from __future__ import annotations

from dgen.dialects import builtin, llvm
from dgen.dialects.builtin import StaticString
from toy.dialects import affine


class AffineToLLVMLowering:
    def __init__(self):
        self.loop_counter = 0
        self.ops: list[builtin.Op] = []
        self.value_map: dict[builtin.Value, builtin.Value] = {}  # affine -> llvm
        self.alloc_shapes: dict[builtin.Value, list[int]] = {}  # llvm alloca -> shape
        self.alloc_sizes: dict[builtin.Value, int] = {}  # llvm alloca -> total size
        self.current_label = StaticString("entry")

    def lower_module(self, m: builtin.Module) -> builtin.Module:
        functions = [self.lower_function(f) for f in m.functions]
        return builtin.Module(functions=functions)

    def lower_function(self, f: builtin.FuncOp) -> builtin.FuncOp:
        self.loop_counter = 0
        self.ops = []
        self.value_map = {}
        self.alloc_shapes = {}
        self.alloc_sizes = {}
        self.current_label = StaticString("entry")

        for op in f.body.ops:
            self.lower_op(op)

        ops = self.ops
        self.ops = []
        return builtin.FuncOp(
            name=f.name,
            body=builtin.Block(ops=ops),
            func_type=builtin.Function(result=builtin.Nil()),
        )

    def _map(self, old: builtin.Value) -> builtin.Value:
        """Resolve an affine Value to its LLVM counterpart."""
        return self.value_map.get(old, old)

    def lower_op(self, op: builtin.Op):
        if isinstance(op, affine.AllocOp):
            self._lower_alloc(op)
        elif isinstance(op, affine.DeallocOp):
            pass  # Stack alloc, no free needed
        elif isinstance(op, affine.LoadOp):
            self._lower_load(op)
        elif isinstance(op, affine.StoreOp):
            self._lower_store(op)
        elif isinstance(op, affine.ForOp):
            self._lower_for(op)
        elif isinstance(op, builtin.ConstantOp):
            new_op = builtin.ConstantOp(value=op.value, type=op.type)
            self.ops.append(new_op)
            self.value_map[op] = new_op
        elif isinstance(op, affine.ArithMulFOp):
            llvm_op = llvm.FMulOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            self.ops.append(llvm_op)
            self.value_map[op] = llvm_op
        elif isinstance(op, affine.ArithAddFOp):
            llvm_op = llvm.FAddOp(lhs=self._map(op.lhs), rhs=self._map(op.rhs))
            self.ops.append(llvm_op)
            self.value_map[op] = llvm_op
        elif isinstance(op, affine.PrintOp):
            self._lower_print(op)
        elif isinstance(op, builtin.ReturnOp):
            val = self._map(op.value) if op.value is not None else None
            self.ops.append(builtin.ReturnOp(value=val))

    def _lower_alloc(self, op: affine.AllocOp):
        total = 1
        for d in op.shape:
            total *= d
        alloca_op = llvm.AllocaOp(elem_count=total)
        self.ops.append(alloca_op)
        self.value_map[op] = alloca_op
        self.alloc_shapes[alloca_op] = list(op.shape)
        self.alloc_sizes[alloca_op] = total

    def _lower_load(self, op: affine.LoadOp):
        memref_val = self._map(op.memref)
        index_vals = [self._map(v) for v in op.indices]
        linear = self._linearize_indices(memref_val, index_vals)
        ptr_op = llvm.GepOp(base=memref_val, index=linear)
        self.ops.append(ptr_op)
        load_op = llvm.LoadOp(ptr=ptr_op)
        self.ops.append(load_op)
        self.value_map[op] = load_op

    def _lower_store(self, op: affine.StoreOp):
        memref_val = self._map(op.memref)
        index_vals = [self._map(v) for v in op.indices]
        linear = self._linearize_indices(memref_val, index_vals)
        ptr_op = llvm.GepOp(base=memref_val, index=linear)
        self.ops.append(ptr_op)
        self.ops.append(llvm.StoreOp(value=self._map(op.value), ptr=ptr_op))

    def _linearize_indices(
        self, memref: builtin.Value, indices: list[builtin.Value]
    ) -> builtin.Value:
        if len(indices) == 1:
            return indices[0]

        shape = self.alloc_shapes[memref]

        result_val: builtin.Value | None = None
        for i, idx_val in enumerate(indices):
            # Compute stride: product of shape[i+1], shape[i+2], ...
            stride = 1
            for j in range(i + 1, len(shape)):
                stride *= shape[j]

            if stride == 1:
                if result_val is None:
                    result_val = idx_val
                else:
                    add_op = llvm.AddOp(lhs=result_val, rhs=idx_val)
                    self.ops.append(add_op)
                    result_val = add_op
            else:
                stride_op = builtin.ConstantOp(value=stride, type=builtin.IndexType())
                self.ops.append(stride_op)
                mul_op = llvm.MulOp(lhs=idx_val, rhs=stride_op)
                self.ops.append(mul_op)
                if result_val is None:
                    result_val = mul_op
                else:
                    add_op = llvm.AddOp(lhs=result_val, rhs=mul_op)
                    self.ops.append(add_op)
                    result_val = add_op

        assert result_val is not None
        return result_val

    def _lower_for(self, op: affine.ForOp):
        loop_id = self.loop_counter
        self.loop_counter += 1

        header_label = StaticString(f"loop_header{loop_id}")
        body_label = StaticString(f"loop_body{loop_id}")
        exit_label = StaticString(f"loop_exit{loop_id}")

        # Init: constant lo, br header
        init_op = builtin.ConstantOp(value=op.lo, type=builtin.IndexType())
        self.ops.append(init_op)
        prev_label = self.current_label
        self.ops.append(llvm.BrOp(dest=header_label))

        # Header label
        self.ops.append(llvm.LabelOp(label_name=header_label))
        self.current_label = header_label

        # Phi node for loop variable (back-edge value patched after body)
        phi_op = llvm.PhiOp(
            values=[init_op, None],  # type: ignore[list-item]  # patched after body
            labels=[prev_label, body_label],
        )
        self.ops.append(phi_op)
        # Map the affine loop variable to the LLVM phi node
        self.value_map[op.body.args[0]] = phi_op

        # Compare and branch
        hi_op = builtin.ConstantOp(value=op.hi, type=builtin.IndexType())
        self.ops.append(hi_op)
        cmp_op = llvm.IcmpOp(pred=StaticString("slt"), lhs=phi_op, rhs=hi_op)
        self.ops.append(cmp_op)
        self.ops.append(
            llvm.CondBrOp(cond=cmp_op, true_dest=body_label, false_dest=exit_label)
        )

        # Body label
        self.ops.append(llvm.LabelOp(label_name=body_label))
        self.current_label = body_label

        # Lower body ops
        for child_op in op.body.ops:
            self.lower_op(child_op)

        # Patch phi back-edge label to actual current block
        phi_op.labels[1] = self.current_label

        # Increment and branch back
        one_op = builtin.ConstantOp(value=1, type=builtin.IndexType())
        self.ops.append(one_op)
        next_op = llvm.AddOp(lhs=phi_op, rhs=one_op)
        self.ops.append(next_op)
        # Patch phi back-edge value
        phi_op.values[1] = next_op
        self.ops.append(llvm.BrOp(dest=header_label))

        # Exit label
        self.ops.append(llvm.LabelOp(label_name=exit_label))
        self.current_label = exit_label

    def _lower_print(self, op: affine.PrintOp):
        input_val = self._map(op.input)
        size = self.alloc_sizes[input_val]
        size_op = builtin.ConstantOp(value=size, type=builtin.IndexType())
        self.ops.append(size_op)
        self.ops.append(
            llvm.CallOp(
                callee="print_memref",
                args=[input_val, size_op],
            )
        )


def lower_to_llvm(m: builtin.Module) -> builtin.Module:
    lowering = AffineToLLVMLowering()
    return lowering.lower_module(m)
