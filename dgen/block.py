from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import dgen

from .type import Type
from .type import Value


@dataclass(eq=False, kw_only=True)
class BlockArgument(Value):
    """A block argument (function parameter)."""

    type: Type


@dataclass
class Block:
    ops: list[dgen.Op]
    args: list[BlockArgument] = field(default_factory=list)

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            yield from op.asm
