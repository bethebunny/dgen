from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable, Iterator

from .block import Block
from .dialect import Dialect
from .type import Fields, Type
from .value import Constant, Value


@dataclass(eq=False)
class Op(Value):
    """Base class for all dialect operations."""

    _asm_name: ClassVar[str]
    dialect: ClassVar[Dialect]
    __params__: ClassVar[Fields] = ()
    __operands__: ClassVar[Fields] = ()
    __blocks__: ClassVar[tuple[str, ...]] = ()

    @property
    def operands(self) -> Iterator[tuple[str, Value]]:
        """All Value-typed fields (constant and runtime)."""
        for name, _ in self.__operands__:
            yield name, getattr(self, name)

    @property
    def parameters(self) -> Iterator[tuple[str, Type]]:
        for name, field in self.__params__:
            yield name, getattr(self, name)

    @property
    def blocks(self) -> Iterator[tuple[str, Block]]:
        """All Block-typed fields."""
        for name in self.__blocks__:
            yield name, getattr(self, name)

    @property
    def asm(self) -> Iterable[str]:
        from .asm.formatting import format_func, op_asm

        if self._asm_name == "function":
            return format_func(self)
        return op_asm(self)

    @property
    def ready(self) -> bool:
        return all(
            isinstance(getattr(self, name), Constant) for name, _ in self.__params__
        )
