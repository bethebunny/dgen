"""CLI tool: compile and run C files through the dgen pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import click

from dgen import Dialect
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import Executable, LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.compiler import Compiler
from dgen.passes.control_flow_to_goto import ControlFlowToGoto

from dcc import codegen as _codegen  # noqa: F401 -- registers c.CReturnOp emitter
from dcc.parser.c_parser import parse_c_file
from dcc.parser.lowering import lower
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


def run_file(path: str | Path, args: Sequence[str] = ()) -> object:
    """Compile a .c file and run its last function with int-parsed args."""
    ir = lower(parse_c_file(path))
    exe = c_compiler.compile(ir)
    result = exe.run(*(int(a) for a in args))
    return result.to_json()


@click.command()
@click.argument("source_file", type=click.Path(exists=True))
@click.argument("args", nargs=-1)
def main(source_file: str, args: tuple[str, ...]) -> None:
    """Compile and run a C source file.

    Arguments after SOURCE_FILE are parsed as integers and passed to the
    last function defined in the file.
    """
    click.echo(run_file(source_file, args))


if __name__ == "__main__":
    main()
