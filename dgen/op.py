from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable

from .block import Block
from .dialect import Dialect
from .type import Fields
from .value import Value


@dataclass(eq=False)
class Op(Value):
    """Base class for all dialect operations."""

    _asm_name: ClassVar[str]
    dialect: ClassVar[Dialect]
    __params__: ClassVar[Fields] = ()
    __operands__: ClassVar[Fields] = ()
    __blocks__: ClassVar[tuple[str, ...]] = ()

    @property
    def operands(self) -> Iterable[tuple[str, Value]]:
        """All Value-typed fields (constant and runtime)."""
        for name, _ in self.__params__:
            yield name, getattr(self, name)
        for name, _ in self.__operands__:
            yield name, getattr(self, name)

    @property
    def blocks(self) -> Iterable[tuple[str, Block]]:
        """All Block-typed fields."""
        for name in self.__blocks__:
            yield name, getattr(self, name)

    @property
    def asm(self) -> Iterable[str]:
        from .asm.formatting import op_asm

        return op_asm(self)

    @property
    def ready(self) -> bool:
        return all(
            isinstance(getattr(self, name), Constant) for name, _ in self.__params__
        )
