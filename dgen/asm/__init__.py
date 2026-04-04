"""Shared asm output utilities."""

from collections.abc import Iterable
from typing import Protocol

from .formatting import indent
from .parser import parse_module as parse
from .parser import parse_value


class HasAsm(Protocol):
    @property
    def asm(self) -> Iterable[str]: ...


def format(node: HasAsm) -> str:
    return "\n".join(node.asm)


__all__ = ["HasAsm", "format", "indent", "parse", "parse_value"]
