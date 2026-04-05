"""Shared asm output utilities."""

from collections.abc import Iterable
from typing import Protocol

from .formatting import indent
from .parser import parse, parse_value


class HasAsm(Protocol):
    @property
    def asm(self) -> Iterable[str]: ...


def format(node: HasAsm | object) -> str:
    """Format a value as ASM text.

    For Values, emits dialect ``import`` lines followed by the value's
    ASM. For any other object exposing ``.asm``, concatenates its lines.
    """
    from dgen.module import asm_with_imports
    from dgen.type import Value

    if isinstance(node, Value):
        return "\n".join(asm_with_imports(node))
    return "\n".join(node.asm)


__all__ = ["HasAsm", "format", "indent", "parse", "parse_value"]
