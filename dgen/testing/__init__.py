"""Test helpers for IR assertions."""

from __future__ import annotations

from dgen.codegen import Executable, LLVMCodegen
from dgen.compiler import Compiler
from dgen.ir_diff import structural_diff
from dgen.ir_equiv import graph_equivalent
from dgen.module import Module
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.builtin_to_llvm import BuiltinToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto


def llvm_compile(module: Module) -> Executable:
    """Lower a Module through the standard LLVM pipeline and bundle as an Executable.

    Shortcut for ``Compiler([ControlFlowToGoto, BuiltinToLLVM, AlgebraToLLVM],
    LLVMCodegen()).run(module)`` — the pass set that ``dgen.codegen.compile``
    used to hardcode before it moved out of codegen.
    """
    return Compiler(
        [ControlFlowToGoto(), BuiltinToLLVM(), AlgebraToLLVM()],
        LLVMCodegen(),
    ).run(module)


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


def assert_ir_equivalent(actual: Module, expected: Module) -> None:
    """Assert that two IR modules are graph-equivalent.

    Compares use-def graph structure via Merkle fingerprinting. Passes if the
    two modules compute the same thing, regardless of op ordering or SSA names.
    On failure, shows a semantic diff.
    """
    if not graph_equivalent(actual, expected):
        raise AssertionError(structural_diff(actual, expected))
