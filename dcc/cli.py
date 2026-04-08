"""CLI tool: compile and run C files through the dgen pipeline."""

from __future__ import annotations

from pathlib import Path

import click

from dgen import Dialect
from dgen.codegen import Executable, LLVMCodegen
from dgen.compiler import Compiler
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.builtin_to_llvm import BuiltinToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.memory_to_llvm import MemoryToLLVM

from dcc.passes.c_lvalue_to_memory import CLvalueToMemory

# Make dcc dialects discoverable for IR parsing round-trips.
Dialect.paths.append(Path(__file__).parent / "dialects")

# Pass ordering: lvalue elimination must precede memory-to-LLVM.
# See dcc/docs/plans/c-frontend-redesign.md for full rationale.
c_compiler: Compiler[Executable] = Compiler(
    passes=[
        # CStructLayout(),         # Brick 8: before lvalue elimination
        # CImplicitConversions(),  # Brick 9: before lvalue elimination
        CLvalueToMemory(),
        # CToLLVM(),               # Brick 10: after lvalue elimination
        ControlFlowToGoto(),
        MemoryToLLVM(),
        BuiltinToLLVM(),
        AlgebraToLLVM(),
    ],
    exit=LLVMCodegen(),
)


@click.command()
@click.argument("source_file", type=click.Path(exists=True))
@click.argument("args", nargs=-1)
def main(source_file: str, args: tuple[str, ...]) -> None:
    """Compile and run a C source file."""
    raise NotImplementedError("Full CLI not yet implemented (Brick 4+)")


if __name__ == "__main__":
    main()
