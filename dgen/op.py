from __future__ import annotations

import dataclasses
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

    @property
    def operands(self) -> Iterable[tuple[str, Value]]:
        """All Value-typed fields."""
        for f in dataclasses.fields(self):
            if isinstance(attr := getattr(self, f.name), Value):
                yield f.name, attr

    @property
    def blocks(self) -> Iterable[tuple[str, Block]]:
        """All Block-typed fields."""
        for f in dataclasses.fields(self):
            if isinstance(attr := getattr(self, f.name), Block):
                yield f.name, attr

    @property
    def asm(self) -> Iterable[str]:
        from .asm.formatting import op_asm

        return op_asm(self)
