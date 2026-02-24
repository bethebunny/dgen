"""CLI tool: compile and run .toy files."""

from pathlib import Path

import click

from dgen.dialects import builtin
from dgen.staging import compile_and_run_staged
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.affine_to_llvm import lower_to_llvm
from toy.passes.optimize import optimize
from toy.passes.shape_inference import infer_shapes
from toy.passes.toy_to_affine import lower_to_affine


def _lower(m: builtin.Module) -> builtin.Module:
    return lower_to_llvm(lower_to_affine(m))


def _parse_arg(arg: str) -> object:
    """Parse a string arg to a Python value via ASM expr parser."""
    from dgen.asm.parser import IRParser, parse_expr

    return parse_expr(IRParser(arg))


def run(source: str, *, args: list | None = None) -> object:
    """Compile and run a .toy source string through the full pipeline."""
    ast = parse_toy(source)
    ir = lower(ast)
    # Parse string args (from CLI) to Python values, set parameter types
    if args:
        from toy.dialects.affine import shape_memory
        from toy.dialects.toy import TensorType

        args = [_parse_arg(a) if isinstance(a, str) else a for a in args]
        func = ir.functions[0]
        for arg, param in zip(args, func.body.args):
            if isinstance(arg, list):
                param.type = TensorType(shape=shape_memory([len(arg)]))
    opt = optimize(ir)
    return compile_and_run_staged(
        opt,
        infer=infer_shapes,
        lower=_lower,
        args=args,
    )


@click.command()
@click.argument("source_file", type=click.Path(exists=True))
@click.argument("args", nargs=-1)
def main(source_file: str, args: tuple[str, ...]) -> None:
    """Compile and run a .toy source file."""
    run(Path(source_file).read_text(), args=list(args) if args else None)


if __name__ == "__main__":
    main()
