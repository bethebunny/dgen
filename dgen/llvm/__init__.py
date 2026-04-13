"""LLVM backend: lowerings and codegen."""

from __future__ import annotations

from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import Executable, LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.compiler import Compiler


def lower_to_llvm() -> Compiler[Executable]:
    """LLVM lowerings + codegen: memory, builtin, algebra → LLVM IR → JIT."""
    return Compiler(
        passes=[MemoryToLLVM(), BuiltinToLLVM(), AlgebraToLLVM()],
        exit=LLVMCodegen(),
    )
