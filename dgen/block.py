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

    Stores the result value (root of the use-def graph). For blocks whose ops
    form a proper use-def graph, use result= and ops are derived by walk_ops.
    For parser-created blocks with flat op lists that may not form a complete
    graph, use ops= to store the list explicitly. Use result=None for empty
    blocks (e.g. thin label markers).
    """

    result: dgen.Value | None
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
            self._stored_ops: list[dgen.Op] | None = None
        elif ops is not None and ops:
            self.result = ops[-1]
            self._stored_ops = ops
        else:
            # Empty block (no ops, no result — valid for thin label markers)
            self.result = None
            self._stored_ops = None
        self.args = args if args is not None else []

    @property
    def ops(self) -> list[dgen.Op]:
        if self._stored_ops is not None:
            return self._stored_ops
        if self.result is None:
            return []
        return walk_ops(self.result)

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            yield from op.asm
