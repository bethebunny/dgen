"""Ch6: Affine IR to LLVM-like IR lowering."""

from __future__ import annotations

from toy_python.dialects import affine, builtin, llvm


class AffineToLLVMLowering:
    def __init__(self):
        self.counter = 0
        self.loop_counter = 0
        self.ops: list[builtin.Op] = []
        self.alloc_shapes: dict[str, list[int]] = {}
        self.alloc_sizes: dict[str, int] = {}
        self.current_label = "entry"

    def fresh(self) -> str:
        name = f"v{self.counter}"
        self.counter += 1
        return name

    def lower_module(self, m: builtin.Module) -> builtin.Module:
        functions = [self.lower_function(f) for f in m.functions]
        return builtin.Module(functions=functions)

    def lower_function(self, f: builtin.FuncOp) -> builtin.FuncOp:
        self.counter = 0
        self.loop_counter = 0
        self.ops = []
        self.alloc_shapes = {}
        self.alloc_sizes = {}
        self.current_label = "entry"

        for op in f.body.ops:
            self.lower_op(op)

        ops = self.ops
        self.ops = []
        return builtin.FuncOp(
            name=f.name,
            body=builtin.Block(ops=ops),
            func_type=builtin.FuncType(result=builtin.Nil()),
        )

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
        elif isinstance(op, affine.ArithConstantOp):
            self.ops.append(llvm.ConstantOp(result=op.result, value=op.value))
        elif isinstance(op, affine.IndexConstantOp):
            self.ops.append(llvm.IndexConstOp(result=op.result, value=op.value))
        elif isinstance(op, affine.ArithMulFOp):
            self.ops.append(
                llvm.FMulOp(result=op.result, lhs=op.lhs, rhs=op.rhs)
            )
        elif isinstance(op, affine.ArithAddFOp):
            self.ops.append(
                llvm.FAddOp(result=op.result, lhs=op.lhs, rhs=op.rhs)
            )
        elif isinstance(op, affine.PrintOp):
            self._lower_print(op)
        elif isinstance(op, affine.ReturnOp):
            self.ops.append(llvm.ReturnOp(value=op.value))

    def _lower_alloc(self, op: affine.AllocOp):
        total = 1
        for d in op.shape:
            total *= d
        self.ops.append(llvm.AllocaOp(result=op.result, elem_count=total))
        self.alloc_shapes[op.result] = list(op.shape)
        self.alloc_sizes[op.result] = total

    def _lower_load(self, op: affine.LoadOp):
        linear = self._linearize_indices(op.memref, op.indices)
        ptr_name = self.fresh()
        self.ops.append(
            llvm.GepOp(result=ptr_name, base=op.memref, index=linear)
        )
        self.ops.append(llvm.LoadOp(result=op.result, ptr=ptr_name))

    def _lower_store(self, op: affine.StoreOp):
        linear = self._linearize_indices(op.memref, op.indices)
        ptr_name = self.fresh()
        self.ops.append(
            llvm.GepOp(result=ptr_name, base=op.memref, index=linear)
        )
        self.ops.append(llvm.StoreOp(value=op.value, ptr=ptr_name))

    def _linearize_indices(self, memref: str, indices: list[str]) -> str:
        """Linearize multi-dimensional indices into a flat index.

        For shape [d0, d1, ...] and indices [i0, i1, ...]:
        linear = i0 * d1 * d2 * ... + i1 * d2 * ... + iN
        """
        if len(indices) == 1:
            return indices[0]

        shape = self.alloc_shapes[memref]

        result_name = ""
        for i, idx_name in enumerate(indices):
            # Compute stride: product of shape[i+1], shape[i+2], ...
            stride = 1
            for j in range(i + 1, len(shape)):
                stride *= shape[j]

            if stride == 1:
                if not result_name:
                    result_name = idx_name
                else:
                    add_name = self.fresh()
                    self.ops.append(
                        llvm.AddOp(
                            result=add_name, lhs=result_name, rhs=idx_name
                        )
                    )
                    result_name = add_name
            else:
                stride_name = self.fresh()
                self.ops.append(
                    llvm.IndexConstOp(result=stride_name, value=stride)
                )
                mul_name = self.fresh()
                self.ops.append(
                    llvm.MulOp(result=mul_name, lhs=idx_name, rhs=stride_name)
                )
                if not result_name:
                    result_name = mul_name
                else:
                    add_name = self.fresh()
                    self.ops.append(
                        llvm.AddOp(
                            result=add_name, lhs=result_name, rhs=mul_name
                        )
                    )
                    result_name = add_name

        return result_name

    def _lower_for(self, op: affine.ForOp):
        loop_id = self.loop_counter
        self.loop_counter += 1

        header_label = f"loop_header{loop_id}"
        body_label = f"loop_body{loop_id}"
        exit_label = f"loop_exit{loop_id}"
        var_name = op.var_name

        # Init: iconst lo, br header
        init_name = self.fresh()
        self.ops.append(llvm.IndexConstOp(result=init_name, value=op.lo))
        prev_label = self.current_label
        self.ops.append(llvm.BrOp(dest=header_label))

        # Header label
        self.ops.append(llvm.LabelOp(name=header_label))
        self.current_label = header_label

        # Phi node for loop variable (back-edge label patched after body)
        next_name = self.fresh()
        phi_op = llvm.PhiOp(
            result=var_name,
            values=[init_name, next_name],
            labels=[prev_label, body_label],  # back-edge label patched after body
        )
        self.ops.append(phi_op)

        # Compare and branch
        hi_name = self.fresh()
        self.ops.append(llvm.IndexConstOp(result=hi_name, value=op.hi))
        cmp_name = self.fresh()
        self.ops.append(
            llvm.IcmpOp(
                result=cmp_name, pred="slt", lhs=var_name, rhs=hi_name
            )
        )
        self.ops.append(
            llvm.CondBrOp(
                cond=cmp_name, true_dest=body_label, false_dest=exit_label
            )
        )

        # Body label
        self.ops.append(llvm.LabelOp(name=body_label))
        self.current_label = body_label

        # Lower body ops
        for child_op in op.body:
            self.lower_op(child_op)

        # Patch phi back-edge to actual current block (may differ from
        # body_label when the body contains nested loops)
        phi_op.labels[1] = self.current_label

        # Increment and branch back
        one_name = self.fresh()
        self.ops.append(llvm.IndexConstOp(result=one_name, value=1))
        self.ops.append(
            llvm.AddOp(result=next_name, lhs=var_name, rhs=one_name)
        )
        self.ops.append(llvm.BrOp(dest=header_label))

        # Exit label
        self.ops.append(llvm.LabelOp(name=exit_label))
        self.current_label = exit_label

    def _lower_print(self, op: affine.PrintOp):
        input_name = op.input
        size = self.alloc_sizes[input_name]
        size_name = self.fresh()
        self.ops.append(llvm.IndexConstOp(result=size_name, value=size))
        self.ops.append(
            llvm.CallOp(
                result=None,
                callee="print_memref",
                args=[input_name, size_name],
            )
        )


def lower_to_llvm(m: builtin.Module) -> builtin.Module:
    lowering = AffineToLLVMLowering()
    return lowering.lower_module(m)
