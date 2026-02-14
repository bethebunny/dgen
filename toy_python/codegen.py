"""Emit valid LLVM IR text from LLModule and JIT-compile via llvmlite."""

from __future__ import annotations

import ctypes
import sys
from io import StringIO

import llvmlite.binding as llvm

from toy_python.dialects.llvm_ops import (
    LLModule,
    LLFuncOp,
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


# ---------------------------------------------------------------------------
# Runtime: print_memref callback
# ---------------------------------------------------------------------------


@ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64)
def _print_memref(ptr, size):
    arr = (ctypes.c_double * size).from_address(ptr)
    print(", ".join(f"{arr[i]:g}" for i in range(size)))


# ---------------------------------------------------------------------------
# LLVM initialization
# ---------------------------------------------------------------------------

_initialized = False


def _ensure_initialized():
    global _initialized
    if not _initialized:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        llvm.add_symbol(
            "print_memref", ctypes.cast(_print_memref, ctypes.c_void_p).value
        )
        _initialized = True


# ---------------------------------------------------------------------------
# IR emission
# ---------------------------------------------------------------------------


def emit_llvm_ir(module: LLModule) -> str:
    """Emit valid LLVM IR text that llvmlite can parse."""
    lines: list[str] = ["declare void @print_memref(ptr, i64)", ""]
    for func in module.functions:
        lines.extend(_emit_func(func))
    return "\n".join(lines)


def _emit_func(f: LLFuncOp) -> list[str]:
    # Pre-scan: build constants and types maps
    constants: dict[str, str] = {}  # SSA name -> typed literal
    types: dict[str, str] = {}  # SSA name -> LLVM type

    for op in f.body.ops:
        if isinstance(op, LLConstantOp):
            constants[op.result] = f"double {format_float(op.value)}"
            types[op.result] = "double"
        elif isinstance(op, LLIndexConstOp):
            constants[op.result] = f"i64 {op.value}"
            types[op.result] = "i64"
        elif isinstance(op, LLAllocaOp):
            types[op.result] = "ptr"
        elif isinstance(op, LLGepOp):
            types[op.result] = "ptr"
        elif isinstance(op, LLLoadOp):
            types[op.result] = "double"
        elif isinstance(op, (LLFAddOp, LLFMulOp)):
            types[op.result] = "double"
        elif isinstance(op, (LLAddOp, LLMulOp)):
            types[op.result] = "i64"
        elif isinstance(op, LLIcmpOp):
            types[op.result] = "i1"
        elif isinstance(op, LLPhiOp):
            first_val = op.pairs[0].value
            types[op.result] = types.get(first_val, "i64")

    def typed_ref(name: str) -> str:
        """'type value' — e.g. 'double 1.0' or 'ptr %v3'."""
        if name in constants:
            return constants[name]
        return f"{types.get(name, 'i64')} %{name}"

    def bare_ref(name: str) -> str:
        """Just the value — e.g. '1.0' or '%v3'."""
        if name in constants:
            return constants[name].split(" ", 1)[1]
        return f"%{name}"

    lines = [f"define void @{f.name}() {{", "entry:"]

    for op in f.body.ops:
        if isinstance(op, (LLConstantOp, LLIndexConstOp)):
            continue  # inlined at use sites

        if isinstance(op, LLLabelOp):
            lines.append(f"{op.name}:")
        elif isinstance(op, LLAllocaOp):
            lines.append(
                f"  %{op.result} = alloca double, i64 {op.elem_count}"
            )
        elif isinstance(op, LLGepOp):
            lines.append(
                f"  %{op.result} = getelementptr double, ptr %{op.base},"
                f" {typed_ref(op.index)}"
            )
        elif isinstance(op, LLLoadOp):
            lines.append(
                f"  %{op.result} = load double, {typed_ref(op.ptr)}"
            )
        elif isinstance(op, LLStoreOp):
            lines.append(
                f"  store {typed_ref(op.value)}, {typed_ref(op.ptr)}"
            )
        elif isinstance(op, LLFAddOp):
            lines.append(
                f"  %{op.result} = fadd double {bare_ref(op.lhs)},"
                f" {bare_ref(op.rhs)}"
            )
        elif isinstance(op, LLFMulOp):
            lines.append(
                f"  %{op.result} = fmul double {bare_ref(op.lhs)},"
                f" {bare_ref(op.rhs)}"
            )
        elif isinstance(op, LLAddOp):
            lines.append(
                f"  %{op.result} = add i64 {bare_ref(op.lhs)},"
                f" {bare_ref(op.rhs)}"
            )
        elif isinstance(op, LLMulOp):
            lines.append(
                f"  %{op.result} = mul i64 {bare_ref(op.lhs)},"
                f" {bare_ref(op.rhs)}"
            )
        elif isinstance(op, LLIcmpOp):
            lines.append(
                f"  %{op.result} = icmp {op.pred} i64 {bare_ref(op.lhs)},"
                f" {bare_ref(op.rhs)}"
            )
        elif isinstance(op, LLBrOp):
            lines.append(f"  br label %{op.dest}")
        elif isinstance(op, LLCondBrOp):
            lines.append(
                f"  br i1 %{op.cond}, label %{op.true_dest},"
                f" label %{op.false_dest}"
            )
        elif isinstance(op, LLPhiOp):
            ty = types.get(op.result, "i64")
            pairs = ", ".join(
                f"[ {bare_ref(p.value)}, %{p.label} ]" for p in op.pairs
            )
            lines.append(f"  %{op.result} = phi {ty} {pairs}")
        elif isinstance(op, LLCallOp):
            if op.callee == "print_memref" and len(op.args) == 2:
                a = f"{typed_ref(op.args[0])}, {typed_ref(op.args[1])}"
                lines.append(f"  call void @print_memref({a})")
            else:
                a = ", ".join(typed_ref(arg) for arg in op.args)
                if op.result is not None:
                    lines.append(
                        f"  %{op.result} = call void @{op.callee}({a})"
                    )
                else:
                    lines.append(f"  call void @{op.callee}({a})")
        elif isinstance(op, LLReturnOp):
            if op.value is not None:
                lines.append(f"  ret {typed_ref(op.value)}")
            else:
                lines.append("  ret void")

    lines.append("}")
    return lines


# ---------------------------------------------------------------------------
# JIT compilation
# ---------------------------------------------------------------------------


def compile_and_run(
    ll_module: LLModule, capture_output: bool = False
) -> str | None:
    """Emit LLVM IR, JIT-compile, and execute the module's main function."""
    _ensure_initialized()
    ir_text = emit_llvm_ir(ll_module)

    mod = llvm.parse_assembly(ir_text)
    mod.verify()

    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    engine = llvm.create_mcjit_compiler(mod, target_machine)

    func_ptr = engine.get_function_address("main")
    cfunc = ctypes.CFUNCTYPE(None)(func_ptr)

    if capture_output:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            cfunc()
            return sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
    else:
        cfunc()
        return None
