"""Fingerprint-guided structural diff for IR modules.

Reports semantic differences between two modules: ops present in actual with
no matching fingerprint in expected (added), and ops present in expected with
no matching fingerprint in actual (removed or changed).

For the common 1:1 case — one removed, one added — both forms are shown
side-by-side as a single "changed" entry.
"""

from __future__ import annotations

from collections import Counter

import dgen
from dgen import asm
from dgen.asm.formatting import SlotTracker, op_asm
from dgen.dialects.builtin import FunctionOp
from dgen.ir_equiv import Fingerprinter
from dgen.module import Module


def diff_modules(actual: Module, expected: Module) -> str:
    """Return a human-readable diff between two modules, keyed on semantics."""
    actual_funcs = {f.name: f for f in actual.functions}
    expected_funcs = {f.name: f for f in expected.functions}

    sections: list[str] = ["IR equivalence check failed."]

    for name in sorted(expected_funcs.keys() | actual_funcs.keys()):
        if name not in actual_funcs:
            func_text = "\n".join(
                f"-   {line}" for line in asm.format(expected_funcs[name]).splitlines()
            )
            sections.append(f"function '{name}': missing entirely\n{func_text}")
        elif name not in expected_funcs:
            func_text = "\n".join(
                f"+   {line}" for line in asm.format(actual_funcs[name]).splitlines()
            )
            sections.append(f"function '{name}': unexpected\n{func_text}")
        else:
            diff = _diff_function(actual_funcs[name], expected_funcs[name])
            if diff:
                sections.append(f"function '{name}':\n{diff}")

    return "\n\n".join(sections)


def _diff_function(actual_func: FunctionOp, expected_func: FunctionOp) -> str | None:
    """Diff two functions at the op level. Returns diff text or None if equivalent."""
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

    only_in_actual = _unmatched(actual_ops, actual_fps, Counter(expected_fps))
    only_in_expected = _unmatched(expected_ops, expected_fps, Counter(actual_fps))

    if not only_in_actual and not only_in_expected:
        return None

    lines: list[str] = []

    if len(only_in_actual) == len(only_in_expected) == 1:
        # 1:1 case: show as a single changed op side-by-side
        lines.append("  changed:")
        for line in op_asm(only_in_expected[0], tracker_e):
            lines.append(f"-   {line}")
        for line in op_asm(only_in_actual[0], tracker_a):
            lines.append(f"+   {line}")
    else:
        if only_in_expected:
            lines.append("  missing from actual:")
            for op in only_in_expected:
                for line in op_asm(op, tracker_e):
                    lines.append(f"-   {line}")
        if only_in_actual:
            lines.append("  unexpected in actual:")
            for op in only_in_actual:
                for line in op_asm(op, tracker_a):
                    lines.append(f"+   {line}")

    return "\n".join(lines)


def _unmatched(
    ops: list[dgen.Op], fps: list[bytes], available: Counter[bytes]
) -> list[dgen.Op]:
    """Return ops whose fingerprint has no remaining match in available."""
    result = []
    for op, fp in zip(ops, fps):
        if available[fp] > 0:
            available[fp] -= 1
        else:
            result.append(op)
    return result
