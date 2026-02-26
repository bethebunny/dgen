from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable

from .block import Block
from .dialect import Dialect
from .value import Value


@dataclass(eq=False)
class Op(Value):
    """Base class for all dialect operations."""

    _asm_name: ClassVar[str]
    dialect: ClassVar[Dialect]
    __arg_fields__: ClassVar[tuple[str, ...]] = ()
    __has_body__: ClassVar[bool] = False

    @property
    def operands(self) -> Iterable[tuple[str, Value]]:
        """All Value-typed arg fields."""
        for name in self.__arg_fields__:
            if isinstance(attr := getattr(self, name), Value):
                yield name, attr

    @property
    def blocks(self) -> Iterable[tuple[str, Block]]:
        """All Block-typed fields."""
        if self.__has_body__:
            yield "body", getattr(self, "body")

    @property
    def asm(self) -> Iterable[str]:
        from .asm.formatting import op_asm

        return op_asm(self)
