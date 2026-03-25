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
    def ready(self) -> bool:
        return False

    @property
    def __constant__(self) -> Memory:
        raise TypeError(f"BlockArgument %{self.name} is not a constant")


class Block:
    """A block of ops with arguments, parameters, and captures.

    Ops are derived by walking the use-def graph from the result value.

    ``parameters`` are bound once by the op's lowering pass (e.g. ``%self``
    for loop-header labels); callers never pass them explicitly.
    ``args`` are passed by callers at every call/branch site (phi nodes).
    ``captures`` are outer-scope values referenced directly; they are leaves
    in walk_ops (the walk stops at capture boundaries) and do not generate
    phi nodes.
    """

    result: dgen.Value
    args: list[BlockArgument]
    parameters: list[BlockArgument]
    captures: list[dgen.Value]

    def __init__(
        self,
        *,
        result: dgen.Value,
        args: list[BlockArgument] | None = None,
        parameters: list[BlockArgument] | None = None,
        captures: list[dgen.Value] | None = None,
    ) -> None:
        self.result = result
        self.args = args if args is not None else []
        self.parameters = parameters if parameters is not None else []
        self.captures = captures if captures is not None else []

    @property
    def ops(self) -> list[dgen.Op]:
        return walk_ops(self.result, stop=set(self.captures) if self.captures else None)

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            yield from op.asm
