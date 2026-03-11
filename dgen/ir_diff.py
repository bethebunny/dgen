"""Fingerprint-guided unified diff for IR modules.

Matches ops across two IR modules using Merkle fingerprints (order- and
label-insensitive), then emits a unified diff showing only semantic changes
with configurable context lines.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterator

from dgen import asm
from dgen.asm.formatting import SlotTracker, op_asm
from dgen.dialects.builtin import FunctionOp
from dgen.ir_equiv import Fingerprinter
from dgen.module import Module


def diff_modules(actual: Module, expected: Module, context: int = 3) -> str:
    """Return a unified-diff-style string comparing two modules semantically."""
    actual_funcs = {f.name: f for f in actual.functions}
    expected_funcs = {f.name: f for f in expected.functions}

    diffs: list[str] = []

    for name in sorted(expected_funcs.keys() | actual_funcs.keys()):
        if name not in actual_funcs:
            lines = "\n".join(
                f"-  {line}" for line in asm.format(expected_funcs[name]).splitlines()
            )
            diffs.append(f"function '{name}':\n{lines}")
        elif name not in expected_funcs:
            lines = "\n".join(
                f"+  {line}" for line in asm.format(actual_funcs[name]).splitlines()
            )
            diffs.append(f"function '{name}':\n{lines}")
        else:
            body = "\n".join(
                _diff_function(actual_funcs[name], expected_funcs[name], context)
            )
            if body:
                diffs.append(f"function '{name}':\n{body}")

    if not diffs:
        return ""
    return "\n\n".join(["IR equivalence check failed.", *diffs])


def _diff_function(
    actual_func: FunctionOp, expected_func: FunctionOp, context: int
) -> Iterator[str]:
    """Yield unified-diff lines comparing two function bodies."""
    fp_a = Fingerprinter()
    fp_e = Fingerprinter()
    for _, block in actual_func.blocks:
        fp_a.register_block(block)
    for _, block in expected_func.blocks:
        fp_e.register_block(block)

    tracker_a = SlotTracker()
    tracker_a.register([actual_func])
    tracker_e = SlotTracker()
    tracker_e.register([expected_func])

    actual_ops = list(actual_func.body.ops)
    expected_ops = list(expected_func.body.ops)

    actual_fps = [fp_a.fingerprint(op) for op in actual_ops]
    expected_fps = [fp_e.fingerprint(op) for op in expected_ops]

    if actual_fps == expected_fps:
        return

    actual_fmt = [list(op_asm(op, tracker_a)) for op in actual_ops]
    expected_fmt = [list(op_asm(op, tracker_e)) for op in expected_ops]

    exp_starts = _line_starts(expected_fmt)
    act_starts = _line_starts(actual_fmt)

    matcher = difflib.SequenceMatcher(None, expected_fps, actual_fps, autojunk=False)

    for group in matcher.get_grouped_opcodes(n=context):
        ei1, ei2 = group[0][1], group[-1][2]
        ai1, ai2 = group[0][3], group[-1][4]

        exp_start = exp_starts[ei1] if ei1 < len(exp_starts) else 1
        exp_count = sum(len(expected_fmt[k]) for k in range(ei1, ei2))
        act_start = act_starts[ai1] if ai1 < len(act_starts) else 1
        act_count = sum(len(actual_fmt[k]) for k in range(ai1, ai2))

        yield f"@@ -{exp_start},{exp_count} +{act_start},{act_count} @@"

        for tag, pi1, pi2, qi1, qi2 in group:
            if tag == "equal":
                for k in range(pi1, pi2):
                    for line in expected_fmt[k]:
                        yield f"   {line}"
            elif tag == "delete":
                for k in range(pi1, pi2):
                    for line in expected_fmt[k]:
                        yield f"-  {line}"
            elif tag == "insert":
                for k in range(qi1, qi2):
                    for line in actual_fmt[k]:
                        yield f"+  {line}"
            elif tag == "replace":
                for k in range(pi1, pi2):
                    for line in expected_fmt[k]:
                        yield f"-  {line}"
                for k in range(qi1, qi2):
                    for line in actual_fmt[k]:
                        yield f"+  {line}"


def _line_starts(formatted_ops: list[list[str]]) -> list[int]:
    """Return 1-indexed text-line start position for each op."""
    starts: list[int] = []
    total = 1
    for lines in formatted_ops:
        starts.append(total)
        total += len(lines)
    return starts
