"""CLI tool: compile and run .toy files."""

from pathlib import Path

import click

from dgen.staging import compile_and_run_staged
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.affine_to_llvm import lower_to_llvm
from toy.passes.optimize import optimize
from toy.passes.shape_inference import infer_shapes
from toy.passes.toy_to_affine import lower_to_affine


def _lower(m):
    return lower_to_llvm(lower_to_affine(m))


def run(
    source: str, *, args: list | None = None, capture_output: bool = False
) -> str | None:
    """Compile and run a .toy source string through the full pipeline."""
    ast = parse_toy(source)
    ir = lower(ast)
    # Set parameter types from runtime args (needed for stage-1 staging)
    if args:
        from toy.dialects.toy import TensorType

        func = ir.functions[0]
        for arg_val, param in zip(args, func.body.args):
            if isinstance(arg_val, list):
                param.type = TensorType(shape=[len(arg_val)])
    opt = optimize(ir)
    return compile_and_run_staged(
        opt,
        infer=infer_shapes,
        lower=_lower,
        args=args,
        capture_output=capture_output,
    )


def run_ir(
    ir_text: str, *, args: list | None = None, capture_output: bool = False
) -> str | None:
    """Parse IR text and run through the staging pipeline."""
    from dgen.asm.parser import IRParser

    module = IRParser(ir_text).parse_module()
    return compile_and_run_staged(
        module,
        infer=infer_shapes,
        lower=_lower,
        args=args,
        capture_output=capture_output,
    )


@click.command()
@click.argument("source_file", type=click.Path(exists=True))
def main(source_file):
    """Compile and run a .toy source file."""
    run(Path(source_file).read_text())


if __name__ == "__main__":
    main()
