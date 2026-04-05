"""Unified diff for IR values.

``graph_equivalent`` collapses all SSA-name and op-order differences to a
single bytes comparison, so if two roots disagree the remaining diff *is*
structural — a plain textual unified-diff over ``asm.format`` output is
enough.

CLI usage:
    python -m dgen.ir_diff expected.ir actual.ir
    python -m dgen.ir_diff expected.ir actual.ir -C 5
    python -m dgen.ir_diff expected.ir actual.ir -I toy/dialects
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import click

import dgen
from dgen import Dialect, asm
from dgen.asm.parser import parse
from dgen.ir_equiv import graph_equivalent


def structural_diff(actual: dgen.Value, expected: dgen.Value) -> str:
    """Return a human-readable diff between two IRs."""
    return diff_values(actual, expected)


def diff_values(
    actual: dgen.Value | None,
    expected: dgen.Value | None,
    context: int = 3,
) -> str:
    """Return a unified diff comparing two IR values.

    Returns ``""`` if the two values are graph-equivalent (same IR up to
    op ordering and SSA renaming). Otherwise, returns a unified diff of
    their ``asm.format`` output with the usual ``---``/``+++`` headers.
    """
    if actual is None and expected is None:
        return ""
    if actual is None:
        assert expected is not None
        lines = "\n".join(f"-{line}" for line in asm.format(expected).splitlines())
        return "--- expected\n+++ actual\n" + lines
    if expected is None:
        lines = "\n".join(f"+{line}" for line in asm.format(actual).splitlines())
        return "--- expected\n+++ actual\n" + lines

    if graph_equivalent(actual, expected):
        return ""

    diff = difflib.unified_diff(
        asm.format(expected).splitlines(),
        asm.format(actual).splitlines(),
        fromfile="expected",
        tofile="actual",
        n=context,
        lineterm="",
    )
    return "\n".join(diff)


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
    "-I",
    "--include",
    "include_dirs",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory to search for .dgen dialect files (repeatable).",
)
def diff(
    expected: click.utils.LazyFile,
    actual: click.utils.LazyFile,
    context: int,
    color: bool | None,
    include_dirs: tuple[Path, ...],
) -> None:
    """Semantically compare two IR files.

    Differences are reported as a unified diff keyed on Merkle fingerprints,
    so op ordering and SSA label renaming are ignored.  Exits 0 if equivalent,
    1 if different.

    Non-core dialects are discovered via --include:

        python -m dgen.ir_diff expected.ir actual.ir \\
            -I toy/dialects -I actor/dialects
    """
    for d in include_dirs:
        Dialect.paths.append(d)

    expected_value = parse(expected.read())
    actual_value = parse(actual.read())

    diff_text = diff_values(actual_value, expected_value, context=context)
    if not diff_text:
        return

    for line in diff_text.splitlines():
        if line.startswith("+"):
            click.echo(click.style(line, fg="green"), color=color)
        elif line.startswith("-"):
            click.echo(click.style(line, fg="red"), color=color)
        elif line.startswith("@@"):
            click.echo(click.style(line, fg="cyan"), color=color)
        else:
            click.echo(line)

    sys.exit(1)


if __name__ == "__main__":
    diff()
