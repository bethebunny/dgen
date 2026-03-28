from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable, Iterator

from .block import Block
from .dialect import Dialect
from .type import Fields, Value


@dataclass(eq=False)
class Op(Value):
    """Base class for all dialect operations."""

    name: str | None = None
    asm_name: ClassVar[str]
    dialect: ClassVar[Dialect]
    __operands__: ClassVar[Fields] = ()
    __blocks__: ClassVar[tuple[str, ...]] = ()

    @property
    def operands(self) -> Iterator[tuple[str, Value]]:
        """All Value-typed fields (constant and runtime)."""
        for name, _ in self.__operands__:
            yield name, getattr(self, name)

    def replace_operand(self, old: Value, new: Value) -> None:
        """Replace all occurrences of old with new in operand fields."""
        for name, _ in self.__operands__:
            val = getattr(self, name)
            if val is old:
                setattr(self, name, new)

    @property
    def blocks(self) -> Iterator[tuple[str, Block]]:
        """All Block-typed fields."""
        for name in self.__blocks__:
            yield name, getattr(self, name)

    @property
    def dependencies(self) -> Iterator[Value]:
        """All Value dependencies for use-def graph traversal."""
        for _, operand in self.operands:
            if isinstance(operand, Value):
                yield operand
        for _, param in self.parameters:
            if isinstance(param, list):
                for item in param:
                    if isinstance(item, Value):
                        yield item
            elif isinstance(param, Value):
                yield param
        yield self.type
        for _, block in self.blocks:
            yield from block.captures
            for p in block.parameters:
                yield p.type
            for a in block.args:
                yield a.type

    @property
    def asm(self) -> Iterable[str]:
        from .asm.formatting import op_asm

        return op_asm(self)
