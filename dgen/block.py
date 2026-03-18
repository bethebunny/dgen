from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import dgen

from .graph import walk_ops
from .type import Memory, TypeType, Value


@dataclass(eq=False, kw_only=True)
class BlockArgument(Value):
    """A block argument (function parameter)."""

    name: str | None = None
    type: Value[TypeType]

    @property
    def __constant__(self) -> Memory:
        raise TypeError(f"BlockArgument %{self.name} is not a constant")


class Block:
    """A block of ops with arguments.

    Ops are derived by walking the use-def graph from the result value.
    """

    result: dgen.Value
    args: list[BlockArgument]

    def __init__(
        self,
        *,
        result: dgen.Value,
        args: list[BlockArgument] | None = None,
    ) -> None:
        self.result = result
        self.args = args if args is not None else []

    @property
    def ops(self) -> list[dgen.Op]:
        return walk_ops(self.result)

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            yield from op.asm
