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

    Stores the result value (root of the use-def graph). When constructed
    with result=, ops are derived by walking the use-def graph. When
    constructed with ops=, the provided list is used directly until
    _stored_ops is cleared (e.g. after a pass runs).
    """

    result: dgen.Value
    args: list[BlockArgument]

    def __init__(
        self,
        *,
        result: dgen.Value | None = None,
        ops: list[dgen.Op] | None = None,
        args: list[BlockArgument] | None = None,
    ) -> None:
        if result is not None:
            self.result = result
            self._stored_ops: list[dgen.Op] | None = ops
        elif ops is not None and ops:
            self.result = ops[-1]
            self._stored_ops = ops
        else:
            raise ValueError("Block needs either result= or non-empty ops=")
        self.args = args if args is not None else []

    @property
    def ops(self) -> list[dgen.Op]:
        if self._stored_ops is not None:
            return self._stored_ops
        return walk_ops(self.result)

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            yield from op.asm
