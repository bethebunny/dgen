"""Ch6: LLVM-like IR text serialization."""

from toy.llvm_ir import (
    LLModule, LLFuncOp, LLBlock, AnyLLVMOp,
    LLAllocaOp, LLGepOp, LLLoadOp, LLStoreOp,
    LLFAddOp, LLFMulOp, LLConstantOp, LLIndexConstOp,
    LLAddOp, LLMulOp, LLIcmpOp,
    LLBrOp, LLCondBrOp, LLLabelOp, LLPhiOp,
    LLCallOp, LLReturnOp, PhiPair,
)
from collections import Optional


fn print_llvm_module(m: LLModule) -> String:
    var s = String("")
    for i in range(len(m.functions)):
        if i > 0:
            s += "\n"
        s += _print_func(m.functions[i])
    return s


fn _print_func(f: LLFuncOp) -> String:
    var s = String("define void @" + f.name + "():\n")
    for i in range(len(f.body.ops)):
        s += _print_op(f.body.ops[i])
    return s


fn _print_op(op: AnyLLVMOp) -> String:
    if op.isa[LLAllocaOp]():
        return "    %" + String(op[LLAllocaOp].result) + " = alloca f64, " + String(op[LLAllocaOp].elem_count) + "\n"
    if op.isa[LLGepOp]():
        return "    %" + String(op[LLGepOp].result) + " = gep %" + String(op[LLGepOp].base) + ", %" + String(op[LLGepOp].index) + "\n"
    if op.isa[LLLoadOp]():
        return "    %" + String(op[LLLoadOp].result) + " = load %" + String(op[LLLoadOp].ptr) + "\n"
    if op.isa[LLStoreOp]():
        return "    store %" + String(op[LLStoreOp].value) + ", %" + String(op[LLStoreOp].ptr) + "\n"
    if op.isa[LLFAddOp]():
        return "    %" + String(op[LLFAddOp].result) + " = fadd %" + String(op[LLFAddOp].lhs) + ", %" + String(op[LLFAddOp].rhs) + "\n"
    if op.isa[LLFMulOp]():
        return "    %" + String(op[LLFMulOp].result) + " = fmul %" + String(op[LLFMulOp].lhs) + ", %" + String(op[LLFMulOp].rhs) + "\n"
    if op.isa[LLConstantOp]():
        return "    %" + String(op[LLConstantOp].result) + " = fconst " + _format_float(op[LLConstantOp].value) + "\n"
    if op.isa[LLIndexConstOp]():
        return "    %" + String(op[LLIndexConstOp].result) + " = iconst " + String(op[LLIndexConstOp].value) + "\n"
    if op.isa[LLAddOp]():
        return "    %" + String(op[LLAddOp].result) + " = add %" + String(op[LLAddOp].lhs) + ", %" + String(op[LLAddOp].rhs) + "\n"
    if op.isa[LLMulOp]():
        return "    %" + String(op[LLMulOp].result) + " = mul %" + String(op[LLMulOp].lhs) + ", %" + String(op[LLMulOp].rhs) + "\n"
    if op.isa[LLIcmpOp]():
        return "    %" + String(op[LLIcmpOp].result) + " = icmp " + String(op[LLIcmpOp].pred) + " %" + String(op[LLIcmpOp].lhs) + ", %" + String(op[LLIcmpOp].rhs) + "\n"
    if op.isa[LLBrOp]():
        return "    br " + String(op[LLBrOp].dest) + "\n"
    if op.isa[LLCondBrOp]():
        return "    cond_br %" + String(op[LLCondBrOp].cond) + ", " + String(op[LLCondBrOp].true_dest) + ", " + String(op[LLCondBrOp].false_dest) + "\n"
    if op.isa[LLLabelOp]():
        return String(op[LLLabelOp].name) + ":\n"
    if op.isa[LLPhiOp]():
        var s = String("    %" + String(op[LLPhiOp].result) + " = phi")
        for j in range(len(op[LLPhiOp].pairs)):
            var pair = op[LLPhiOp].pairs[j].copy()
            s += " [%" + String(pair.value) + ", " + String(pair.label) + "]"
        s += "\n"
        return s
    if op.isa[LLCallOp]():
        var s = String("    ")
        if op[LLCallOp].result:
            s += "%" + String(op[LLCallOp].result.value()) + " = "
        s += "call @" + String(op[LLCallOp].callee) + "("
        for j in range(len(op[LLCallOp].args)):
            if j > 0:
                s += ", "
            s += "%" + String(op[LLCallOp].args[j])
        s += ")\n"
        return s
    if op.isa[LLReturnOp]():
        if op[LLReturnOp].value:
            return "    ret %" + String(op[LLReturnOp].value.value()) + "\n"
        return "    ret void\n"
    return "    <unknown llvm op>\n"


fn _format_float(v: Float64) -> String:
    var iv = Int(v)
    if Float64(iv) == v:
        return String(iv) + ".0"
    return String(v)
