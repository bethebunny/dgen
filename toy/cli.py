"""CLI tool: compile and run .toy files."""

from collections.abc import Sequence
from pathlib import Path

import ast

import click

from dgen.module import Module
from dgen.passes.pass_manager import PassManager
from dgen.staging import compile_and_run_staged
from toy.dialects import shape_constant
from toy.dialects.toy import Tensor
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.affine_to_llvm import AffineToLLVMLowering
from toy.passes.optimize import optimize
from toy.passes.shape_inference import infer_shapes
from toy.passes.toy_to_affine import ToyToAffine


def _lower(m: Module) -> Module:
    return PassManager([ToyToAffine(), AffineToLLVMLowering()]).run(m)


def _parse_arg(arg: str) -> object:
    """Parse a string arg to a Python value."""
    return ast.literal_eval(arg)


def _set_param_types(ir: Module, args: Sequence[object]) -> None:
    """Set function parameter types from runtime argument values.

    For list arguments, sets the parameter type to a 1-D Tensor
    with the list's length as the shape dimension.
    """
    func = ir.functions[0]
    for arg, param in zip(args, func.body.args):
        if isinstance(arg, list):
            param.type = Tensor(shape=shape_constant([len(arg)]))


def run(source: str, *, args: Sequence[object | str] = ()) -> object:
    """Compile and run a .toy source string through the full pipeline."""
    ast = parse_toy(source)
    ir = lower(ast)
    parsed_args = [_parse_arg(a) if isinstance(a, str) else a for a in args]
    if parsed_args:
        _set_param_types(ir, parsed_args)
    opt = optimize(ir)
    return compile_and_run_staged(
        opt,
        infer=infer_shapes,
        lower=_lower,
        args=parsed_args,
    )


@click.command()
@click.argument("source_file", type=click.Path(exists=True))
@click.argument("args", nargs=-1)
def main(source_file: str, args: tuple[str, ...]) -> None:
    """Compile and run a .toy source file."""
    run(Path(source_file).read_text(), args=args)


if __name__ == "__main__":
    main()
