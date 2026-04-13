"""dgen: compile and execute a .dgen.asm file with JSON-encoded arguments.

Reads a textual IR file, compiles its top-level value through the default
LLVM pipeline, runs it with arguments parsed from JSON, and prints the
result as JSON. Argument and return-value marshalling is fully driven by
each type's ``__layout__`` — no per-type CLI code.

If the top-level value is a function, the JSON arguments are passed to it.
Otherwise the value is wrapped in a no-arg function and evaluated as-is.

    python -m dgen path/to/program.dgen.asm '8' '42'
"""

from __future__ import annotations

import json
from pathlib import Path

import click

import dgen
from dgen.asm.parser import parse
from dgen.builtins import pack
from dgen.dialects.function import Function, FunctionOp
from dgen.llvm import lower_to_llvm
from dgen.passes import lower_builtin_dialects
from dgen.passes.compiler import Compiler
from dgen.type import format_value


def _default_compiler() -> Compiler:
    return Compiler(
        passes=[lower_builtin_dialects()],
        exit=lower_to_llvm(),
    )


def _wrap_in_function(value: dgen.Value) -> FunctionOp:
    """Wrap a non-function value in a no-arg function returning it."""
    return FunctionOp(
        name="main",
        body=dgen.Block(result=value),
        result_type=value.type,
        type=Function(arguments=pack([]), result_type=value.type),
    )


@click.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.argument("args", nargs=-1)
def main(source: Path, args: tuple[str, ...]) -> None:
    """Compile and run a .dgen.asm file."""
    value = parse(source.read_text())
    if not isinstance(value, FunctionOp):
        value = _wrap_in_function(value)
    exe = _default_compiler().compile(value)
    parsed_args = [json.loads(a) for a in args]
    result = exe.run(*parsed_args)
    click.echo(format_value(result.to_native_value()))


if __name__ == "__main__":
    main()
