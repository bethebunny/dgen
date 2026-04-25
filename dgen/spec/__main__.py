"""CLI: python -m dgen.spec path/to/dialect.dgen"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import click

import dgen  # noqa: F401 — importing dgen installs the .dgen import hook
from dgen.spec.stubs import generate_pyi


def _path_to_module(dgen_path: Path) -> str | None:
    """Convert an absolute ``.dgen`` file path to a dotted Python module name
    by finding a covering ``sys.path`` entry."""
    resolved = dgen_path.resolve()
    for entry in sys.path:
        try:
            rel = resolved.relative_to(Path(entry).resolve())
        except ValueError:
            continue
        return ".".join(rel.with_suffix("").parts)
    return None


@click.command()
@click.argument("dgen_file", type=click.Path(exists=True, path_type=Path))
def main(dgen_file: Path) -> None:
    """Generate a .pyi type stub for a .dgen dialect file."""
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
