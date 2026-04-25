"""CLI tool: compile and run .toy files."""

from collections.abc import Sequence
from pathlib import Path

import ast

import click

import dgen
from dgen.llvm import lower_to_llvm
from dgen.llvm.codegen import Executable
from dgen.passes.compiler import Compiler
from dgen.dialects.function import FunctionOp
from dgen.passes import lower_builtin_dialects
from toy.dialects import shape_constant
from toy.dialects.toy import Tensor
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes import lower_toy

# Make toy dialects discoverable for IR parsing round-trips.
dgen.PATH.append(Path(__file__).parent / "dialects")

toy_compiler: Compiler[Executable] = Compiler(
    passes=[lower_toy(), lower_builtin_dialects()],
    exit=lower_to_llvm(),
)


def _parse_arg(arg: str) -> object:
    """Parse a string arg to a Python value."""
    return ast.literal_eval(arg)


def _set_param_types(func: FunctionOp, args: Sequence[object]) -> None:
    """Set function parameter types from runtime argument values.

    For list arguments, sets the parameter type to a 1-D Tensor
    with the list's length as the shape dimension.
    """
    for arg, param in zip(args, func.body.args):
        if isinstance(arg, list):
            param.type = Tensor(shape=shape_constant([len(arg)]))


def run(source: str, *, args: Sequence[object | str] = ()) -> object:
    """Compile and run a .toy source string through the full pipeline."""
    ast_node = parse_toy(source)
    ir = lower(ast_node)
    parsed_args = [_parse_arg(a) if isinstance(a, str) else a for a in args]
    if parsed_args:
        _set_param_types(ir, parsed_args)
    exe = toy_compiler.compile(ir)
    return exe.run(*parsed_args)


@click.command()
@click.argument("source_file", type=click.Path(exists=True))
@click.argument("args", nargs=-1)
def main(source_file: str, args: tuple[str, ...]) -> None:
    """Compile and run a .toy source file."""
    run(Path(source_file).read_text(), args=args)


if __name__ == "__main__":
    main()
