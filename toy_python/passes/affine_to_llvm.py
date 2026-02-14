"""Ch6: Affine IR to LLVM-like IR lowering."""

from __future__ import annotations

from toy_python.dialects.affine import (
    AffineModule,
    AffineFuncOp,
    AnyAffineOp,
    AllocOp,
    DeallocOp,
    AffineLoadOp,
    AffineStoreOp,
    AffineForOp,
    ArithConstantOp,
    IndexConstantOp,
    ArithMulFOp,
    ArithAddFOp,
    AffinePrintOp,
    AffineReturnOp,
)
from toy_python.dialects.llvm import (
    LLModule,
    LLFuncOp,
    LLBlock,
    AnyLLVMOp,
    LLAllocaOp,
    LLGepOp,
    LLLoadOp,
    LLStoreOp,
    LLFAddOp,
    LLFMulOp,
    LLConstantOp,
    LLIndexConstOp,
    LLAddOp,
    LLMulOp,
    LLIcmpOp,
    LLBrOp,
    LLCondBrOp,
    LLLabelOp,
    LLPhiOp,
    PhiPair,
    LLCallOp,
    LLReturnOp,
)


class AffineToLLVMLowering:
    def __init__(self):
        self.counter = 0
        self.loop_counter = 0
        self.ops: list[AnyLLVMOp] = []
        self.alloc_shapes: dict[str, list[int]] = {}
        self.alloc_sizes: dict[str, int] = {}
        self.current_label = "entry"

    def fresh(self) -> str:
        name = f"v{self.counter}"
        self.counter += 1
        return name

    def lower_module(self, m: AffineModule) -> LLModule:
        functions = [self.lower_function(f) for f in m.functions]
        return LLModule(functions=functions)

    def lower_function(self, f: AffineFuncOp) -> LLFuncOp:
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
        return LLFuncOp(name=f.name, body=LLBlock(ops=ops))

    def lower_op(self, op: AnyAffineOp):
        if isinstance(op, AllocOp):
            self._lower_alloc(op)
        elif isinstance(op, DeallocOp):
            pass  # Stack alloc, no free needed
        elif isinstance(op, AffineLoadOp):
            self._lower_load(op)
        elif isinstance(op, AffineStoreOp):
            self._lower_store(op)
        elif isinstance(op, AffineForOp):
            self._lower_for(op)
        elif isinstance(op, ArithConstantOp):
            self.ops.append(LLConstantOp(result=op.result, value=op.value))
        elif isinstance(op, IndexConstantOp):
            self.ops.append(LLIndexConstOp(result=op.result, value=op.value))
        elif isinstance(op, ArithMulFOp):
            self.ops.append(
                LLFMulOp(result=op.result, lhs=op.lhs, rhs=op.rhs)
            )
        elif isinstance(op, ArithAddFOp):
            self.ops.append(
                LLFAddOp(result=op.result, lhs=op.lhs, rhs=op.rhs)
            )
        elif isinstance(op, AffinePrintOp):
            self._lower_print(op)
        elif isinstance(op, AffineReturnOp):
            self.ops.append(LLReturnOp(value=op.value))

    def _lower_alloc(self, op: AllocOp):
        total = 1
        for d in op.shape:
            total *= d
        self.ops.append(LLAllocaOp(result=op.result, elem_count=total))
        self.alloc_shapes[op.result] = list(op.shape)
        self.alloc_sizes[op.result] = total

    def _lower_load(self, op: AffineLoadOp):
        linear = self._linearize_indices(op.memref, op.indices)
        ptr_name = self.fresh()
        self.ops.append(
            LLGepOp(result=ptr_name, base=op.memref, index=linear)
        )
        self.ops.append(LLLoadOp(result=op.result, ptr=ptr_name))

    def _lower_store(self, op: AffineStoreOp):
        linear = self._linearize_indices(op.memref, op.indices)
        ptr_name = self.fresh()
        self.ops.append(
            LLGepOp(result=ptr_name, base=op.memref, index=linear)
        )
        self.ops.append(LLStoreOp(value=op.value, ptr=ptr_name))

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
                        LLAddOp(
                            result=add_name, lhs=result_name, rhs=idx_name
                        )
                    )
                    result_name = add_name
            else:
                stride_name = self.fresh()
                self.ops.append(
                    LLIndexConstOp(result=stride_name, value=stride)
                )
                mul_name = self.fresh()
                self.ops.append(
                    LLMulOp(result=mul_name, lhs=idx_name, rhs=stride_name)
                )
                if not result_name:
                    result_name = mul_name
                else:
                    add_name = self.fresh()
                    self.ops.append(
                        LLAddOp(
                            result=add_name, lhs=result_name, rhs=mul_name
                        )
                    )
                    result_name = add_name

        return result_name

    def _lower_for(self, op: AffineForOp):
        loop_id = self.loop_counter
        self.loop_counter += 1

        header_label = f"loop_header{loop_id}"
        body_label = f"loop_body{loop_id}"
        exit_label = f"loop_exit{loop_id}"
        var_name = op.var_name

        # Init: iconst lo, br header
        init_name = self.fresh()
        self.ops.append(LLIndexConstOp(result=init_name, value=op.lo))
        prev_label = self.current_label
        self.ops.append(LLBrOp(dest=header_label))

        # Header label
        self.ops.append(LLLabelOp(name=header_label))
        self.current_label = header_label

        # Phi node for loop variable (back-edge label patched after body)
        next_name = self.fresh()
        phi_op = LLPhiOp(
            result=var_name,
            pairs=[
                PhiPair(value=init_name, label=prev_label),
                PhiPair(value=next_name, label=body_label),  # placeholder
            ],
        )
        self.ops.append(phi_op)

        # Compare and branch
        hi_name = self.fresh()
        self.ops.append(LLIndexConstOp(result=hi_name, value=op.hi))
        cmp_name = self.fresh()
        self.ops.append(
            LLIcmpOp(
                result=cmp_name, pred="slt", lhs=var_name, rhs=hi_name
            )
        )
        self.ops.append(
            LLCondBrOp(
                cond=cmp_name, true_dest=body_label, false_dest=exit_label
            )
        )

        # Body label
        self.ops.append(LLLabelOp(name=body_label))
        self.current_label = body_label

        # Lower body ops
        for child_op in op.body:
            self.lower_op(child_op)

        # Patch phi back-edge to actual current block (may differ from
        # body_label when the body contains nested loops)
        phi_op.pairs[1] = PhiPair(value=next_name, label=self.current_label)

        # Increment and branch back
        one_name = self.fresh()
        self.ops.append(LLIndexConstOp(result=one_name, value=1))
        self.ops.append(
            LLAddOp(result=next_name, lhs=var_name, rhs=one_name)
        )
        self.ops.append(LLBrOp(dest=header_label))

        # Exit label
        self.ops.append(LLLabelOp(name=exit_label))
        self.current_label = exit_label

    def _lower_print(self, op: AffinePrintOp):
        input_name = op.input
        size = self.alloc_sizes[input_name]
        size_name = self.fresh()
        self.ops.append(LLIndexConstOp(result=size_name, value=size))
        self.ops.append(
            LLCallOp(
                result=None,
                callee="print_memref",
                args=[input_name, size_name],
            )
        )


def lower_to_llvm(m: AffineModule) -> LLModule:
    lowering = AffineToLLVMLowering()
    return lowering.lower_module(m)
