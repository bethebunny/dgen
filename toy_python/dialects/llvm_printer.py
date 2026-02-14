"""Ch6: LLVM-like IR text serialization."""

from toy_python.dialects.llvm_ops import (
    LLModule,
    LLFuncOp,
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
    LLCallOp,
    LLReturnOp,
    format_float,
)


def print_llvm_module(m: LLModule) -> str:
    s = ""
    for i, func in enumerate(m.functions):
        if i > 0:
            s += "\n"
        s += _print_func(func)
    return s


def _print_func(f: LLFuncOp) -> str:
    s = f"define void @{f.name}():\n"
    for op in f.body.ops:
        s += _print_op(op)
    return s


def _print_op(op: AnyLLVMOp) -> str:
    if isinstance(op, LLLabelOp):
        return f"{op.name}:\n"
    if isinstance(op, LLAllocaOp):
        return f"    %{op.result} = alloca f64, {op.elem_count}\n"
    if isinstance(op, LLGepOp):
        return f"    %{op.result} = gep %{op.base}, %{op.index}\n"
    if isinstance(op, LLLoadOp):
        return f"    %{op.result} = load %{op.ptr}\n"
    if isinstance(op, LLStoreOp):
        return f"    store %{op.value}, %{op.ptr}\n"
    if isinstance(op, LLFAddOp):
        return f"    %{op.result} = fadd %{op.lhs}, %{op.rhs}\n"
    if isinstance(op, LLFMulOp):
        return f"    %{op.result} = fmul %{op.lhs}, %{op.rhs}\n"
    if isinstance(op, LLConstantOp):
        return f"    %{op.result} = fconst {format_float(op.value)}\n"
    if isinstance(op, LLIndexConstOp):
        return f"    %{op.result} = iconst {op.value}\n"
    if isinstance(op, LLAddOp):
        return f"    %{op.result} = add %{op.lhs}, %{op.rhs}\n"
    if isinstance(op, LLMulOp):
        return f"    %{op.result} = mul %{op.lhs}, %{op.rhs}\n"
    if isinstance(op, LLIcmpOp):
        return f"    %{op.result} = icmp {op.pred} %{op.lhs}, %{op.rhs}\n"
    if isinstance(op, LLBrOp):
        return f"    br {op.dest}\n"
    if isinstance(op, LLCondBrOp):
        return f"    cond_br %{op.cond}, {op.true_dest}, {op.false_dest}\n"
    if isinstance(op, LLPhiOp):
        pairs = " ".join(f"[%{p.value}, {p.label}]" for p in op.pairs)
        return f"    %{op.result} = phi {pairs}\n"
    if isinstance(op, LLCallOp):
        args_str = ", ".join(f"%{a}" for a in op.args)
        if op.result is not None:
            return f"    %{op.result} = call @{op.callee}({args_str})\n"
        return f"    call @{op.callee}({args_str})\n"
    if isinstance(op, LLReturnOp):
        if op.value is not None:
            return f"    ret %{op.value}\n"
        return "    ret void\n"
    return "    <unknown llvm op>\n"
