"""CLI: python -m dgen.gen path/to/dialect.dgen"""

from pathlib import Path

import click

from dgen.gen.parser import parse
from dgen.gen.python import generate


def _parse_import(value: str) -> tuple[str, str]:
    key, _, path = value.partition("=")
    if not path:
        raise click.BadParameter(f"expected module=python.path, got {value!r}")
    return key, path


@click.command()
@click.argument("dgen_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--import",
    "-I",
    "imports",
    multiple=True,
    help="Map a .dgen module to a Python path: builtin=dgen.dialects.builtin",
)
def main(dgen_file: Path, imports: tuple[str, ...]) -> None:
    """Generate Python dialect code from a .dgen file."""
    import_map = dict(_parse_import(i) for i in imports)
    source = dgen_file.read_text()
    ast = parse(source)
    dialect_name = dgen_file.stem
    code = generate(ast, dialect_name=dialect_name, import_map=import_map)
    click.echo(code)


if __name__ == "__main__":
    main()
