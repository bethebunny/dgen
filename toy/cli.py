"""CLI tool: compile and run .toy files."""

from pathlib import Path

import click

from dgen.codegen import compile_and_run
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.affine_to_llvm import lower_to_llvm
from toy.passes.optimize import optimize
from toy.passes.shape_inference import infer_shapes
from toy.passes.toy_to_affine import lower_to_affine


@click.command()
@click.argument("source_file", type=click.Path(exists=True))
def main(source_file):
    """Compile and run a .toy source file."""
    source = Path(source_file).read_text()
    ast = parse_toy(source)
    ir = lower(ast)
    opt = optimize(ir)
    typed = infer_shapes(opt)
    affine = lower_to_affine(typed)
    ll = lower_to_llvm(affine)
    compile_and_run(ll)


if __name__ == "__main__":
    main()
