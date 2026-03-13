"""Test helpers for IR assertions."""

from __future__ import annotations

from dgen.ir_diff import structural_diff
from dgen.ir_equiv import graph_equivalent
from dgen.module import Module


def assert_ir_equivalent(actual: Module, expected: Module) -> None:
    """Assert that two IR modules are graph-equivalent.

    Compares use-def graph structure via Merkle fingerprinting. Passes if the
    two modules compute the same thing, regardless of op ordering or SSA names.
    On failure, shows a semantic diff.
    """
    if not graph_equivalent(actual, expected):
        raise AssertionError(structural_diff(actual, expected))
