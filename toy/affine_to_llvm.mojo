"""Ch6: Affine IR to LLVM-like IR lowering."""

from collections import Optional, Dict

from toy.affine import (
    AffineModule, AffineFuncOp, AffineBlock, AnyAffineOp,
    AllocOp, DeallocOp, AffineLoadOp, AffineStoreOp,
    AffineForOp, ArithConstantOp, IndexConstantOp,
    ArithMulFOp, ArithAddFOp, AffinePrintOp, AffineReturnOp,
)
from toy.llvm_ir import (
    LLModule, LLFuncOp, LLBlock, AnyLLVMOp,
    LLAllocaOp, LLGepOp, LLLoadOp, LLStoreOp,
    LLFAddOp, LLFMulOp, LLConstantOp, LLIndexConstOp,
    LLAddOp, LLMulOp, LLIcmpOp,
    LLBrOp, LLCondBrOp, LLLabelOp, LLPhiOp, PhiPair,
    LLCallOp, LLReturnOp,
)


struct AffineToLLVMLowering(Movable):
    var counter: Int
    var loop_counter: Int
    var ops: List[AnyLLVMOp]
    var alloc_shapes: Dict[String, List[Int]]
    var alloc_sizes: Dict[String, Int]
    var current_label: String

    fn __init__(out self):
        self.counter = 0
        self.loop_counter = 0
        self.ops = List[AnyLLVMOp]()
        self.alloc_shapes = Dict[String, List[Int]]()
        self.alloc_sizes = Dict[String, Int]()
        self.current_label = "entry"

    fn fresh(mut self) -> String:
        var name = String("v") + String(self.counter)
        self.counter += 1
        return name^

    fn lower_module(mut self, m: AffineModule) raises -> LLModule:
        var functions = List[LLFuncOp]()
        for i in range(len(m.functions)):
            functions.append(self.lower_function(m.functions[i]))
        return LLModule(functions=functions^)

    fn lower_function(mut self, f: AffineFuncOp) raises -> LLFuncOp:
        self.counter = 0
        self.loop_counter = 0
        self.ops = List[AnyLLVMOp]()
        self.alloc_shapes = Dict[String, List[Int]]()
        self.alloc_sizes = Dict[String, Int]()
        self.current_label = "entry"

        for i in range(len(f.body.ops)):
            self.lower_op(f.body.ops[i])

        var ops = self.ops^
        self.ops = List[AnyLLVMOp]()
        return LLFuncOp(
            name=String(f.name),
            body=LLBlock(ops=ops^),
        )

    fn lower_op(mut self, op: AnyAffineOp) raises:
        if op.isa[AllocOp]():
            self._lower_alloc(op[AllocOp])
        elif op.isa[DeallocOp]():
            pass  # Stack alloc, no free needed
        elif op.isa[AffineLoadOp]():
            self._lower_load(op[AffineLoadOp])
        elif op.isa[AffineStoreOp]():
            self._lower_store(op[AffineStoreOp])
        elif op.isa[AffineForOp]():
            self._lower_for(op[AffineForOp])
        elif op.isa[ArithConstantOp]():
            self.ops.append(AnyLLVMOp(LLConstantOp(
                result=String(op[ArithConstantOp].result),
                value=op[ArithConstantOp].value,
            )))
        elif op.isa[IndexConstantOp]():
            self.ops.append(AnyLLVMOp(LLIndexConstOp(
                result=String(op[IndexConstantOp].result),
                value=op[IndexConstantOp].value,
            )))
        elif op.isa[ArithMulFOp]():
            self.ops.append(AnyLLVMOp(LLFMulOp(
                result=String(op[ArithMulFOp].result),
                lhs=String(op[ArithMulFOp].lhs),
                rhs=String(op[ArithMulFOp].rhs),
            )))
        elif op.isa[ArithAddFOp]():
            self.ops.append(AnyLLVMOp(LLFAddOp(
                result=String(op[ArithAddFOp].result),
                lhs=String(op[ArithAddFOp].lhs),
                rhs=String(op[ArithAddFOp].rhs),
            )))
        elif op.isa[AffinePrintOp]():
            self._lower_print(op[AffinePrintOp])
        elif op.isa[AffineReturnOp]():
            if op[AffineReturnOp].value:
                self.ops.append(AnyLLVMOp(LLReturnOp(
                    value=String(op[AffineReturnOp].value.value()),
                )))
            else:
                self.ops.append(AnyLLVMOp(LLReturnOp(
                    value=Optional[String](),
                )))

    fn _lower_alloc(mut self, op: AllocOp) raises:
        var total = 1
        for i in range(len(op.shape)):
            total *= op.shape[i]
        self.ops.append(AnyLLVMOp(LLAllocaOp(
            result=String(op.result), elem_count=total,
        )))
        self.alloc_shapes[String(op.result)] = op.shape.copy()
        self.alloc_sizes[String(op.result)] = total

    fn _lower_load(mut self, op: AffineLoadOp) raises:
        var memref = String(op.memref)
        var linear = self._linearize_indices(memref, op.indices)
        var ptr_name = self.fresh()
        self.ops.append(AnyLLVMOp(LLGepOp(
            result=String(ptr_name), base=String(memref), index=String(linear),
        )))
        self.ops.append(AnyLLVMOp(LLLoadOp(
            result=String(op.result), ptr=String(ptr_name),
        )))

    fn _lower_store(mut self, op: AffineStoreOp) raises:
        var memref = String(op.memref)
        var linear = self._linearize_indices(memref, op.indices)
        var ptr_name = self.fresh()
        self.ops.append(AnyLLVMOp(LLGepOp(
            result=String(ptr_name), base=String(memref), index=String(linear),
        )))
        self.ops.append(AnyLLVMOp(LLStoreOp(
            value=String(op.value), ptr=String(ptr_name),
        )))

    fn _linearize_indices(mut self, memref: String, indices: List[String]) raises -> String:
        """Linearize multi-dimensional indices into a flat index.

        For shape [d0, d1, ...] and indices [i0, i1, ...]:
        linear = i0 * d1 * d2 * ... + i1 * d2 * ... + iN
        """
        if len(indices) == 1:
            return String(indices[0])

        var shape = self.alloc_shapes[memref].copy()

        # Compute the linear index
        var result_name = String("")
        for i in range(len(indices)):
            var idx_name = String(indices[i])
            # Compute stride: product of shape[i+1], shape[i+2], ...
            var stride = 1
            for j in range(i + 1, len(shape)):
                stride *= shape[j]

            if stride == 1:
                if len(result_name) == 0:
                    result_name = idx_name
                else:
                    var add_name = self.fresh()
                    self.ops.append(AnyLLVMOp(LLAddOp(
                        result=String(add_name),
                        lhs=String(result_name),
                        rhs=String(idx_name),
                    )))
                    result_name = add_name
            else:
                var stride_name = self.fresh()
                self.ops.append(AnyLLVMOp(LLIndexConstOp(
                    result=String(stride_name), value=stride,
                )))
                var mul_name = self.fresh()
                self.ops.append(AnyLLVMOp(LLMulOp(
                    result=String(mul_name),
                    lhs=String(idx_name),
                    rhs=String(stride_name),
                )))
                if len(result_name) == 0:
                    result_name = mul_name
                else:
                    var add_name = self.fresh()
                    self.ops.append(AnyLLVMOp(LLAddOp(
                        result=String(add_name),
                        lhs=String(result_name),
                        rhs=String(mul_name),
                    )))
                    result_name = add_name

        return result_name^

    fn _lower_for(mut self, op: AffineForOp) raises:
        var loop_id = self.loop_counter
        self.loop_counter += 1

        var header_label = String("loop_header") + String(loop_id)
        var body_label = String("loop_body") + String(loop_id)
        var exit_label = String("loop_exit") + String(loop_id)
        var var_name = String(op.var_name)

        # Init: iconst lo, br header
        var init_name = self.fresh()
        self.ops.append(AnyLLVMOp(LLIndexConstOp(
            result=String(init_name), value=op.lo,
        )))
        var prev_label = String(self.current_label)
        self.ops.append(AnyLLVMOp(LLBrOp(dest=String(header_label))))

        # Header label
        self.ops.append(AnyLLVMOp(LLLabelOp(name=String(header_label))))
        self.current_label = String(header_label)

        # Phi node for loop variable
        var phi_pairs = List[PhiPair]()
        phi_pairs.append(PhiPair(value=String(init_name), label=String(prev_label)))
        var next_name = self.fresh()
        phi_pairs.append(PhiPair(value=String(next_name), label=String(body_label)))
        self.ops.append(AnyLLVMOp(LLPhiOp(
            result=String(var_name), pairs=phi_pairs^,
        )))

        # Compare and branch
        var hi_name = self.fresh()
        self.ops.append(AnyLLVMOp(LLIndexConstOp(
            result=String(hi_name), value=op.hi,
        )))
        var cmp_name = self.fresh()
        self.ops.append(AnyLLVMOp(LLIcmpOp(
            result=String(cmp_name), pred="slt",
            lhs=String(var_name), rhs=String(hi_name),
        )))
        self.ops.append(AnyLLVMOp(LLCondBrOp(
            cond=String(cmp_name),
            true_dest=String(body_label),
            false_dest=String(exit_label),
        )))

        # Body label
        self.ops.append(AnyLLVMOp(LLLabelOp(name=String(body_label))))
        self.current_label = String(body_label)

        # Lower body ops
        for i in range(len(op.body)):
            self.lower_op(op.body[i])

        # Increment and branch back
        var one_name = self.fresh()
        self.ops.append(AnyLLVMOp(LLIndexConstOp(
            result=String(one_name), value=1,
        )))
        self.ops.append(AnyLLVMOp(LLAddOp(
            result=String(next_name),
            lhs=String(var_name),
            rhs=String(one_name),
        )))
        self.ops.append(AnyLLVMOp(LLBrOp(dest=String(header_label))))

        # Exit label
        self.ops.append(AnyLLVMOp(LLLabelOp(name=String(exit_label))))
        self.current_label = String(exit_label)

    fn _lower_print(mut self, op: AffinePrintOp) raises:
        var input_name = String(op.input)
        var size = self.alloc_sizes[input_name]
        var size_name = self.fresh()
        self.ops.append(AnyLLVMOp(LLIndexConstOp(
            result=String(size_name), value=size,
        )))
        var args = List[String]()
        args.append(String(input_name))
        args.append(String(size_name))
        self.ops.append(AnyLLVMOp(LLCallOp(
            result=Optional[String](),
            callee="print_memref",
            args=args^,
        )))


fn lower_to_llvm(m: AffineModule) raises -> LLModule:
    var lowering = AffineToLLVMLowering()
    return lowering.lower_module(m)
