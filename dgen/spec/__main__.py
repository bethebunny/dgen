"""CLI: python -m dgen.spec path/to/dialect.dgen"""

from __future__ import annotations

import importlib
from pathlib import Path

import click

from dgen.spec.importer import _path_to_module, install
from dgen.spec.stubs import generate_pyi


@click.command()
@click.argument("dgen_file", type=click.Path(exists=True, path_type=Path))
def main(dgen_file: Path) -> None:
    """Generate a .pyi type stub for a .dgen dialect file."""
    install()
    module_name = _path_to_module(dgen_file.resolve())
    if module_name is None:
        raise click.ClickException(
            f"Cannot determine module name for {dgen_file}. "
            "Ensure the file is reachable via a sys.path entry "
            "(run from the project root)."
        )
    module = importlib.import_module(module_name)
    click.echo(generate_pyi(module, dialect_name=dgen_file.stem), nl=False)


if __name__ == "__main__":
    main()
