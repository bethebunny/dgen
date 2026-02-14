"""Ch6: LLVM-like IR text serialization."""

from toy.dialects.llvm_ops import (
    LLModule, LLFuncOp, LLBlock, AnyLLVMOp,
    LLAllocaOp, LLGepOp, LLLoadOp, LLStoreOp,
    LLFAddOp, LLFMulOp, LLConstantOp, LLIndexConstOp,
    LLAddOp, LLMulOp, LLIcmpOp,
    LLBrOp, LLCondBrOp, LLLabelOp, LLPhiOp,
    LLCallOp, LLReturnOp,
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
    var s = String("")
    if op.isa[LLLabelOp]():
        op[LLLabelOp].write_asm(s)
        return s + "\n"
    if op.isa[LLAllocaOp]():
        op[LLAllocaOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLGepOp]():
        op[LLGepOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLLoadOp]():
        op[LLLoadOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLStoreOp]():
        op[LLStoreOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLFAddOp]():
        op[LLFAddOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLFMulOp]():
        op[LLFMulOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLConstantOp]():
        op[LLConstantOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLIndexConstOp]():
        op[LLIndexConstOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLAddOp]():
        op[LLAddOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLMulOp]():
        op[LLMulOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLIcmpOp]():
        op[LLIcmpOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLBrOp]():
        op[LLBrOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLCondBrOp]():
        op[LLCondBrOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLPhiOp]():
        op[LLPhiOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLCallOp]():
        op[LLCallOp].write_asm(s)
        return "    " + s + "\n"
    if op.isa[LLReturnOp]():
        op[LLReturnOp].write_asm(s)
        return "    " + s + "\n"
    return "    <unknown llvm op>\n"
