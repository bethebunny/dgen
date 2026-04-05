"""Fingerprint-guided unified diff for IR values.

Matches ops across two IR values using Merkle fingerprints (order- and
label-insensitive), then emits a unified diff showing only semantic changes
with configurable context lines.

CLI usage:
    python -m dgen.ir_diff expected.ir actual.ir
    python -m dgen.ir_diff expected.ir actual.ir -C 5
    python -m dgen.ir_diff expected.ir actual.ir -I toy/dialects
"""

from __future__ import annotations

import difflib
import sys
from collections.abc import Iterator
from pathlib import Path

import click

import dgen
from dgen import Dialect, asm
from dgen.asm.formatting import SlotTracker, _is_sugar_op, op_asm
from dgen.asm.parser import parse
from dgen.graph import all_values, transitive_dependencies
from dgen.ir_equiv import Fingerprinter


def structural_diff(actual: dgen.Value, expected: dgen.Value) -> str:
    """Return a human-readable fingerprint-guided diff between two IRs."""
    return diff_values(actual, expected)


def _make_fingerprinter(root: dgen.Value) -> Fingerprinter:
    fp = Fingerprinter()
    for v in all_values(root):
        for _, block in v.blocks:
            fp.register_block(block)
    return fp


def _top_level_ops(root: dgen.Value) -> list[dgen.Op]:
    """The ops ``asm_with_imports`` would emit as top-level SSA statements."""
    return [
        v
        for v in transitive_dependencies(root)
        if isinstance(v, dgen.Op) and not _is_sugar_op(v)
    ]


def diff_values(
    actual: dgen.Value | None,
    expected: dgen.Value | None,
    context: int = 3,
) -> str:
    """Return a standard unified diff comparing two IR values semantically.

    Uses single-char prefixes (``-``, ``+``, `` ``) and includes
    ``---``/``+++`` file headers so the output can be piped to external
    diff renderers like ``delta``.
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

    fp_a = _make_fingerprinter(actual)
    fp_e = _make_fingerprinter(expected)
    if fp_a.fingerprint(actual) == fp_e.fingerprint(expected):
        return ""

    actual_ops = _top_level_ops(actual)
    expected_ops = _top_level_ops(expected)

    tracker_a = SlotTracker()
    tracker_a.register(actual_ops)
    tracker_e = SlotTracker()
    tracker_e.register(expected_ops)

    actual_fps = [fp_a.fingerprint(op) for op in actual_ops]
    expected_fps = [fp_e.fingerprint(op) for op in expected_ops]

    actual_fmt = [list(op_asm(op, tracker_a)) for op in actual_ops]
    expected_fmt = [list(op_asm(op, tracker_e)) for op in expected_ops]

    body = "\n".join(
        _diff_op_lists(actual_fps, expected_fps, actual_fmt, expected_fmt, context)
    )
    if not body:
        return ""
    return "--- expected\n+++ actual\n" + body


def _diff_op_lists(
    actual_fps: list[bytes],
    expected_fps: list[bytes],
    actual_fmt: list[list[str]],
    expected_fmt: list[list[str]],
    context: int,
) -> Iterator[str]:
    """Yield unified-diff lines from two fingerprint/formatted-op lists."""

    exp_starts = _line_starts(expected_fmt)
    act_starts = _line_starts(actual_fmt)

    matcher = difflib.SequenceMatcher(None, expected_fps, actual_fps, autojunk=False)

    for group in matcher.get_grouped_opcodes(n=context):
        ei1, ei2 = group[0][1], group[-1][2]
        ai1, ai2 = group[0][3], group[-1][4]

        exp_start = exp_starts[ei1] if ei1 < len(exp_starts) else 1
        exp_count = sum(len(lines) for lines in expected_fmt[ei1:ei2])
        act_start = act_starts[ai1] if ai1 < len(act_starts) else 1
        act_count = sum(len(lines) for lines in actual_fmt[ai1:ai2])

        yield f"@@ -{exp_start},{exp_count} +{act_start},{act_count} @@"

        for tag, pi1, pi2, qi1, qi2 in group:
            if tag == "equal":
                for op_lines in expected_fmt[pi1:pi2]:
                    for line in op_lines:
                        yield f" {line}"
            elif tag == "delete":
                for op_lines in expected_fmt[pi1:pi2]:
                    for line in op_lines:
                        yield f"-{line}"
            elif tag == "insert":
                for op_lines in actual_fmt[qi1:qi2]:
                    for line in op_lines:
                        yield f"+{line}"
            elif tag == "replace":
                for op_lines in expected_fmt[pi1:pi2]:
                    for line in op_lines:
                        yield f"-{line}"
                for op_lines in actual_fmt[qi1:qi2]:
                    for line in op_lines:
                        yield f"+{line}"


def _line_starts(formatted_ops: list[list[str]]) -> list[int]:
    """Return 1-indexed text-line start position for each op."""
    starts: list[int] = []
    total = 1
    for lines in formatted_ops:
        starts.append(total)
        total += len(lines)
    return starts


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
