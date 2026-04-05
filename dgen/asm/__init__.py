"""Shared asm output utilities."""

from dgen.module import asm_with_imports
from dgen.type import Value

from .formatting import indent
from .parser import parse


def format(value: Value) -> str:
    """Format a value as ASM text with dialect ``import`` lines."""
    return "\n".join(asm_with_imports(value))


__all__ = ["format", "indent", "parse"]
