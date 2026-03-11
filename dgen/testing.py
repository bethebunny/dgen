"""Test helpers for IR assertions."""

from __future__ import annotations

from dgen.asm.parser import parse_module
from dgen.ir_equiv import graph_equivalent, structural_diff
from dgen.module import Module


def assert_ir_equivalent(actual: Module, expected_ir: str) -> None:
    """Assert that actual is graph-equivalent to the IR described by expected_ir.

    Parses expected_ir and compares use-def graph structure. Passes if the
    two modules compute the same thing, regardless of op ordering or SSA names.
    On failure, shows both formatted IRs side-by-side.
    """
    expected = parse_module(expected_ir)
    if not graph_equivalent(actual, expected):
        raise AssertionError(structural_diff(actual, expected))
