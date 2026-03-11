"""CLI for semantic IR diffing.

Usage:
    python -m dgen.diff_cli expected.ir actual.ir
    python -m dgen.diff_cli expected.ir actual.ir -C 5
    python -m dgen.diff_cli expected.ir actual.ir --color
"""

from __future__ import annotations

import importlib
import sys

import click

from dgen.asm.parser import parse_module
from dgen.ir_diff import diff_modules


@click.command()
@click.argument("expected", type=click.File("r"))
@click.argument("actual", type=click.File("r"))
@click.option(
    "-C",
    "--context",
    default=3,
    show_default=True,
    metavar="N",
    help="Ops of context shown around each change.",
)
@click.option(
    "--color/--no-color",
    default=None,
    help="Force color output on/off (default: auto-detect terminal).",
)
@click.option(
    "-d",
    "--dialect",
    "dialects",
    multiple=True,
    metavar="MODULE",
    help="Python module to import for dialect registration (repeatable).",
)
def diff(
    expected: click.utils.LazyFile,
    actual: click.utils.LazyFile,
    context: int,
    color: bool | None,
    dialects: tuple[str, ...],
) -> None:
    """Semantically compare two IR files.

    Differences are reported as a unified diff keyed on Merkle fingerprints,
    so op ordering and SSA label renaming are ignored.  Exits 0 if equivalent,
    1 if different.

    Dialects used in the IR must be registered first:

        python -m dgen.diff_cli expected.ir actual.ir \\
            -d toy.dialects.toy -d toy.dialects.affine
    """
    for module_path in dialects:
        importlib.import_module(module_path)

    expected_module = parse_module(expected.read())
    actual_module = parse_module(actual.read())

    diff_text = diff_modules(actual_module, expected_module, context=context)
    if not diff_text:
        return

    for line in diff_text.splitlines():
        if line.startswith("+"):
            click.echo(click.style(line, fg="green"), color=color)
        elif line.startswith("-"):
            click.echo(click.style(line, fg="red"), color=color)
        elif line.startswith("@@"):
            click.echo(click.style(line, fg="cyan"), color=color)
        elif line.startswith("IR equivalence"):
            click.echo(click.style(line, bold=True), color=color)
        else:
            click.echo(line)

    sys.exit(1)


if __name__ == "__main__":
    diff()
