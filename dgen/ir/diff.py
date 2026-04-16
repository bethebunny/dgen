"""Semantic diff for IR values.

Matches ops across the two graphs by Merkle fingerprint, aligns the
matched/unmatched ops with a ``SequenceMatcher``, and emits a
unified-diff-shaped listing keyed on per-op declarations. Two roots that
are ``graph_equivalent`` (same IR up to op ordering and SSA renaming)
produce the empty diff; otherwise every emitted line is a genuine
structural change.

CLI usage:
    python -m dgen.ir.diff expected.ir actual.ir
    python -m dgen.ir.diff expected.ir actual.ir -C 5
    python -m dgen.ir.diff expected.ir actual.ir -I toy/dialects
"""

from __future__ import annotations

import difflib
import sys
from collections.abc import Iterator
from pathlib import Path

import click

import dgen
from dgen import Dialect
from dgen.asm.formatting import SlotTracker, _is_sugar_op, op_asm
from dgen.asm.parser import parse
from dgen.ir.traversal import all_values
from dgen.ir.equivalence import Fingerprinter


def structural_diff(actual: dgen.Value, expected: dgen.Value) -> str:
    """Return a human-readable diff between two IRs."""
    return diff_values(actual, expected)


def _reachable_ops(root: dgen.Value) -> list[dgen.Op]:
    """Every reachable Op, in topological order, excluding sugar ops."""
    return [
        v for v in all_values(root) if isinstance(v, dgen.Op) and not _is_sugar_op(v)
    ]


def _register_blocks(fp: Fingerprinter, root: dgen.Value) -> None:
    for v in all_values(root):
        if isinstance(v, dgen.Block):
            fp.register_block(v)
        elif isinstance(v, dgen.Op):
            for _, block in v.blocks:
                fp.register_block(block)


def _op_line(op: dgen.Op, tracker: SlotTracker) -> str:
    """The op's own single-line declaration, without nested block bodies."""
    return next(iter(op_asm(op, tracker, formatted=set())))


def diff_values(
    actual: dgen.Value | None,
    expected: dgen.Value | None,
    context: int = 3,
) -> str:
    """Return a semantic diff comparing two IR values.

    Output is ``""`` iff ``graph_equivalent(actual, expected)``. Otherwise
    every reachable op is fingerprinted, matched across the two sides, and
    emitted as a unified-diff-shaped listing (``---``/``+++`` headers,
    ``@@`` hunks). Since matching is fingerprint-keyed, SSA-name and
    op-ordering differences never appear in the output.
    """
    if actual is None and expected is None:
        return ""
    if actual is None:
        assert expected is not None
        return _one_sided_diff(expected, sign="-")
    if expected is None:
        return _one_sided_diff(actual, sign="+")

    fp_actual = Fingerprinter()
    fp_expected = Fingerprinter()
    _register_blocks(fp_actual, actual)
    _register_blocks(fp_expected, expected)
    if fp_actual.fingerprint(actual) == fp_expected.fingerprint(expected):
        return ""

    actual_ops = _reachable_ops(actual)
    expected_ops = _reachable_ops(expected)
    actual_fps = [fp_actual.fingerprint(op) for op in actual_ops]
    expected_fps = [fp_expected.fingerprint(op) for op in expected_ops]

    tracker_a = SlotTracker()
    tracker_a.register([actual])
    tracker_e = SlotTracker()
    tracker_e.register([expected])
    actual_lines = [_op_line(op, tracker_a) for op in actual_ops]
    expected_lines = [_op_line(op, tracker_e) for op in expected_ops]

    hunks = "\n".join(
        _emit_hunks(expected_fps, actual_fps, expected_lines, actual_lines, context)
    )
    if not hunks:
        return ""
    return "--- expected\n+++ actual\n" + hunks


def _one_sided_diff(value: dgen.Value, *, sign: str) -> str:
    tracker = SlotTracker()
    tracker.register([value])
    ops = _reachable_ops(value)
    body = "\n".join(f"{sign}{_op_line(op, tracker)}" for op in ops)
    count = len(ops)
    header = f"@@ -1,{count} +0,0 @@" if sign == "-" else f"@@ -0,0 +1,{count} @@"
    return f"--- expected\n+++ actual\n{header}\n{body}"


def _emit_hunks(
    expected_fps: list[bytes],
    actual_fps: list[bytes],
    expected_lines: list[str],
    actual_lines: list[str],
    context: int,
) -> Iterator[str]:
    matcher = difflib.SequenceMatcher(None, expected_fps, actual_fps, autojunk=False)
    for group in matcher.get_grouped_opcodes(n=context):
        ei1, ei2 = group[0][1], group[-1][2]
        ai1, ai2 = group[0][3], group[-1][4]
        yield f"@@ -{ei1 + 1},{ei2 - ei1} +{ai1 + 1},{ai2 - ai1} @@"
        for tag, pi1, pi2, qi1, qi2 in group:
            if tag == "equal":
                for line in expected_lines[pi1:pi2]:
                    yield f" {line}"
            else:
                if tag in ("delete", "replace"):
                    for line in expected_lines[pi1:pi2]:
                        yield f"-{line}"
                if tag in ("insert", "replace"):
                    for line in actual_lines[qi1:qi2]:
                        yield f"+{line}"


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

        python -m dgen.ir.diff expected.ir actual.ir \\
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
