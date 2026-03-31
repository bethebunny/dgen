"""Emit valid LLVM IR text from a Module and JIT-compile via llvmlite."""

from dgen.codegen._emit import (
    Executable,
    LLVMCodegen,
    LinearBlock,
    _ctype,
    _emit_func,
    _ensure_initialized,
    _jit_engine,
    _llvm_type,
    compile,
    emit_llvm_ir,
)

__all__ = [
    "Executable",
    "LLVMCodegen",
    "LinearBlock",
    "_ctype",
    "_emit_func",
    "_ensure_initialized",
    "_jit_engine",
    "_llvm_type",
    "compile",
    "emit_llvm_ir",
]
