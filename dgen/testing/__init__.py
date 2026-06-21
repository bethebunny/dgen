"""Test helpers for IR assertions."""

from __future__ import annotations

import llvmlite.binding as llvm_binding

import dgen
from dgen.llvm.codegen import Executable, LLVMCodegen, _ensure_initialized
from dgen.passes.compiler import Compiler
from dgen.ir.diff import structural_diff
from dgen.ir.equivalence import graph_equivalent
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.normalize_region_terminators import NormalizeRegionTerminators


def llvm_compile(value: dgen.Value) -> Executable:
    """Lower a Value through the standard LLVM pipeline and bundle as an Executable.

    Shortcut for ``Compiler([ControlFlowToGoto, NormalizeRegionTerminators,
    BuiltinToLLVM, AlgebraToLLVM], LLVMCodegen()).run(value)`` — the pass
    set that ``dgen.llvm.codegen.compile`` used to hardcode before it
    moved out of codegen.
    """
    return Compiler(
        [
            ControlFlowToGoto(),
            NormalizeRegionTerminators(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        LLVMCodegen(),
    ).run(value)


def assert_valid_llvm(ir: str) -> None:
    """Parse and verify LLVM IR text, raising if it is malformed.

    Text-snapshot comparisons (``assert exe.ir == snapshot``) never parse the
    IR — they only compare strings — so they can silently bless invalid LLVM
    (mismatched phi types, bad arity, ...): a codegen regression that emits
    garbage just gets its snapshot regenerated and "passes". Call this
    alongside such snapshots so malformed IR fails loudly. Mirrors the
    parse+verify that ``_jit_engine`` does on the ``.run()`` path, without
    building a JIT engine.
    """
    _ensure_initialized()
    llvm_binding.parse_assembly(ir).verify()


def strip_prefix(text: str) -> str:
    """Convert a pipe-prefixed multiline string to plain text.

    Each line is stripped of leading whitespace and then:
      - "| content" becomes "content"
      - "|"         becomes ""  (blank line)
      - other       passed through as-is
    A trailing newline is always appended.
    """
    lines = text.strip().splitlines()
    result = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("| "):
            result.append(stripped[2:])
        elif stripped == "|":
            result.append("")
        else:
            result.append(stripped)
    return "\n".join(result) + "\n"


def assert_ir_equivalent(actual: dgen.Value, expected: dgen.Value) -> None:
    """Assert that two IR values are graph-equivalent.

    Compares use-def graph structure via Merkle fingerprinting. Passes if the
    two values compute the same thing, regardless of op ordering or SSA names.
    On failure, shows a semantic diff.
    """
    if not graph_equivalent(actual, expected):
        raise AssertionError(structural_diff(actual, expected))
