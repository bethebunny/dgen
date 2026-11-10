"""CLI: python -m dgen.gen [--target python|mlir] path/to/dialect.dgen"""

from __future__ import annotations

import importlib
from pathlib import Path

import click

from dgen.gen.importer import _path_to_module, install, model_of
from dgen.gen.mlir import generate_td
from dgen.gen.python import generate_pyi


@click.command()
@click.argument("dgen_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--target",
    type=click.Choice(["python", "mlir"]),
    default="python",
    help="python writes a .pyi stub, mlir an ODS .td file.",
)
@click.option(
    "--mlir-prefix",
    default="dgen_",
    help="Starts each MLIR dialect name; use one prefix for every file of a build.",
)
def main(dgen_file: Path, target: str, mlir_prefix: str) -> None:
    """Generate a .pyi type stub or an MLIR ODS file for a .dgen dialect file."""
    install()
    module_name = _path_to_module(dgen_file.resolve())
    if module_name is None:
        raise click.ClickException(
            f"Cannot determine module name for {dgen_file}. "
            "Ensure the file is reachable via a sys.path entry "
            "(run from the project root)."
        )
    module = importlib.import_module(module_name)
    match target:
        case "mlir":
            click.echo(generate_td(model_of(module), prefix=mlir_prefix), nl=False)
        case _:
            click.echo(generate_pyi(module, dialect_name=dgen_file.stem), nl=False)


if __name__ == "__main__":
    main()
