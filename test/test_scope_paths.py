"""Parity spike for docs/plans/scope-paths.md.

``derived_captures`` is the plan's reference implementation of
capture derivation: classify every value reachable from a block's result
against the enclosing scopes' visible sets. If derivation reproduces the
hand-declared captures of real lowered IR, the framework can own capture
authorship (migration step 2 of the plan).
"""

from pathlib import Path

import dgen
from dgen.block import BlockArgument, BlockParameter
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.testing import strip_prefix
from dgen.asm.parser import parse
from dgen.type import is_constant

from dcc.cli import c_compiler
from dcc.parser.c_parser import parse_c_string
from dcc.parser.lowering import lower
from dcc.passes.c_lvalue_to_memory import CLvalueToMemory
from dcc.passes.thread_loop_memory import ThreadLoopMemory


def derived_captures(block: dgen.Block, outer_visible: set) -> set:
    """Reference implementation of scope-path capture derivation."""
    declared = {*block.args, *block.parameters}
    captures: set = set()
    seen: set = set()

    def visit(value: dgen.Value) -> None:
        if value in seen or is_constant(value):
            return
        seen.add(value)
        if value in declared:
            return
        if isinstance(value, (BlockArgument, BlockParameter)):
            captures.add(value)
            return
        if value in outer_visible:
            captures.add(value)
            return
        for dependency in value.dependencies:
            visit(dependency)

    visit(block.result)
    return captures


def assert_capture_parity(root: dgen.Value) -> int:
    """Derive captures for every block reachable from *root* and assert
    equality with the declared lists. Returns how many blocks checked."""
    checked = 0

    def check(block: dgen.Block, outer_visible: set) -> None:
        nonlocal checked
        derived = derived_captures(block, outer_visible)
        declared = set(block.captures)
        assert derived == declared, (
            f"capture parity failed for block of {block.result.name!r}:\n"
            f"  declared - derived: {[v.name for v in declared - derived]}\n"
            f"  derived - declared: {[v.name for v in derived - declared]}"
        )
        checked += 1
        visible = (
            outer_visible
            | set(block.values)
            | {*block.args, *block.parameters}
            | set(block.captures)
        )
        for value in block.values:
            for _, child in value.blocks:
                check(child, visible)

    for _, child in root.blocks:
        check(child, {root})
    return checked


WHILE_LOOP = strip_prefix("""
    | import algebra
    | import number
    | import control_flow
    | import index
    | import record
    | %zero : index.Index = 0
    | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
    |     %ten : index.Index = 10
    |     %cmp : number.Boolean = algebra.less_than(%i, %ten)
    | body(%i: index.Index):
    |     %one : index.Index = 1
    |     %next : index.Index = algebra.add(%i, %one)
    |     %carry : Tuple<[index.Index]> = record.pack([%next])
""")


def test_parity_on_lowered_while():
    lowered = Compiler([ControlFlowToGoto()], IdentityPass()).compile(parse(WHILE_LOOP))
    assert assert_capture_parity(lowered) >= 2


def test_parity_on_lowered_nested_for():
    ir = strip_prefix("""
        | import control_flow
        | import index
        |
        | %outer : Nil = control_flow.for<index.Index(0), index.Index(2)>([]) body(%i: index.Index):
        |     %inner : Nil = control_flow.for<index.Index(0), index.Index(2)>([]) body(%j: index.Index):
        |         %0 : index.Index = 0
        |         %1 : Nil = chain(%0, %0)
    """)
    lowered = Compiler([ControlFlowToGoto()], IdentityPass()).compile(parse(ir))
    assert assert_capture_parity(lowered) >= 4


def test_parity_on_threaded_dcc_loops():
    """The richest capture structure in the tree: dcc's memory-lowered,
    token-threaded nested loops, before goto lowering."""
    ir = lower(
        parse_c_string(
            "int f(int n) { int t = 0; int i = 0;"
            " while (i < n) { int j = 0;"
            " while (j < n) { t = t + 1; j = j + 1; }"
            " i = i + 1; } return t; }"
        )
    )
    threaded = Compiler([CLvalueToMemory(), ThreadLoopMemory()], IdentityPass()).run(ir)
    assert assert_capture_parity(threaded) >= 4


def test_parity_end_to_end_dcc(tmp_path: Path):
    """Full dcc pipeline output (through goto lowering) still derives."""
    ir = lower(
        parse_c_string(
            "int f(int n) { int s = 0; int i = 0;"
            " while (i < n) { if (i - 2 * (i / 2)) { s = s + i; } i = i + 1; }"
            " return s; }"
        )
    )
    exe = c_compiler.compile(ir)
    assert exe.run(5).to_json() == 1 + 3
