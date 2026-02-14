"""Shared asm output utilities."""

from collections.abc import Iterable
from typing import Protocol


class HasAsm(Protocol):
    @property
    def asm(self) -> Iterable[str]: ...


def format(node: HasAsm) -> str:
    return "\n".join(node.asm)


def indent(it: Iterable[str], prefix: str = "    ") -> Iterable[str]:
    for line in it:
        yield f"{prefix}{line}"
