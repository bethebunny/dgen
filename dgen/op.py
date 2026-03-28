from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable

from .dialect import Dialect
from .type import Value


@dataclass(eq=False)
class Op(Value):
    """Base class for all dialect operations."""

    name: str | None = None
    asm_name: ClassVar[str]
    dialect: ClassVar[Dialect]

    def replace_operand(self, old: Value, new: Value) -> None:
        """Replace all occurrences of old with new in operand fields."""
        for name, _ in self.__operands__:
            val = getattr(self, name)
            if val is old:
                setattr(self, name, new)

    @property
    def asm(self) -> Iterable[str]:
        from .asm.formatting import op_asm

        return op_asm(self)
