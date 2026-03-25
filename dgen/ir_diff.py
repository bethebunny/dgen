"""Fingerprint-guided unified diff for IR modules.

Matches ops across two IR modules using Merkle fingerprints (order- and
label-insensitive), then emits a unified diff showing only semantic changes
with configurable context lines.

CLI usage:
    python -m dgen.ir_diff expected.ir actual.ir
    python -m dgen.ir_diff expected.ir actual.ir -C 5
    python -m dgen.ir_diff expected.ir actual.ir --color
"""

from __future__ import annotations

import difflib
import importlib
import sys
from collections.abc import Iterator

import click

from dgen import asm
from dgen.asm.formatting import SlotTracker, op_asm
from dgen.asm.parser import parse_module
from dgen.dialects.function import DefineOp
from dgen.ir_equiv import Fingerprinter
from dgen.module import Module


def structural_diff(actual: Module, expected: Module) -> str:
    """Return a human-readable fingerprint-guided diff between two IRs."""
    return diff_modules(actual, expected)


def diff_modules(actual: Module, expected: Module, context: int = 3) -> str:
    """Return a standard unified diff comparing two modules semantically.

    Uses single-char prefixes (``-``, ``+``, `` ``) and includes
    ``---``/``+++`` file headers so the output can be piped to external
    diff renderers like ``delta``.
    """
    actual_funcs = {f.name: f for f in actual.functions}
    expected_funcs = {f.name: f for f in expected.functions}

    hunks: list[str] = []

    for name in sorted(expected_funcs.keys() | actual_funcs.keys()):
        if name not in actual_funcs:
            lines = "\n".join(
                f"-{line}" for line in asm.format(expected_funcs[name]).splitlines()
            )
            hunks.append(lines)
        elif name not in expected_funcs:
            lines = "\n".join(
                f"+{line}" for line in asm.format(actual_funcs[name]).splitlines()
            )
            hunks.append(lines)
        else:
            body = "\n".join(
                _diff_function(actual_funcs[name], expected_funcs[name], context)
            )
            if body:
                hunks.append(body)

    if not hunks:
        return ""
    return "--- expected\n+++ actual\n" + "\n".join(hunks)


def _diff_function(
    actual_func: DefineOp, expected_func: DefineOp, context: int
) -> Iterator[str]:
    """Yield unified-diff lines comparing two function bodies."""
    fp_a = Fingerprinter()
    fp_e = Fingerprinter()
    for _, block in actual_func.blocks:
        fp_a.register_block(block)
    for _, block in expected_func.blocks:
        fp_e.register_block(block)

    if fp_a.fingerprint(actual_func) == fp_e.fingerprint(expected_func):
        return

    tracker_a = SlotTracker()
    tracker_a.register([actual_func])
    tracker_e = SlotTracker()
    tracker_e.register([expected_func])

    actual_ops = list(actual_func.body.ops)
    expected_ops = list(expected_func.body.ops)

    actual_fps = [fp_a.fingerprint(op) for op in actual_ops]
    expected_fps = [fp_e.fingerprint(op) for op in expected_ops]

    actual_fmt = [list(op_asm(op, tracker_a)) for op in actual_ops]
    expected_fmt = [list(op_asm(op, tracker_e)) for op in expected_ops]

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

        python -m dgen.ir_diff expected.ir actual.ir \\
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
        else:
            click.echo(line)

    sys.exit(1)


if __name__ == "__main__":
    diff()
