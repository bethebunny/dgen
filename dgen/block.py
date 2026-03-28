from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Iterable

import dgen

from .graph import transitive_dependencies
from .type import Memory, TypeType, Value


@dataclass(eq=False, kw_only=True)
class BlockArgument(Value):
    """A runtime block argument — passed by callers at every branch site.

    Block arguments are runtime values (loop induction variables, phi inputs).
    They are never compile-time constants, so ``ready`` is always False.
    """

    name: str | None = None
    type: Value[TypeType]

    @property
    def ready(self) -> bool:
        return False

    @property
    def __constant__(self) -> Memory:
        raise TypeError(f"BlockArgument %{self.name} is not a constant")


@dataclass(eq=False, kw_only=True)
class BlockParameter(Value):
    """A compile-time block parameter — bound once by the lowering pass.

    Block parameters (e.g. ``%self`` on goto label headers) are structural
    values determined at IR construction time. They do not vary at runtime
    and are never passed by callers. Ready when their type is ready.
    """

    name: str | None = None
    type: Value[TypeType]

    @property
    def ready(self) -> bool:
        return self.type.ready

    @property
    def __constant__(self) -> Memory:
        raise TypeError(f"BlockParameter %{self.name} is not a constant")


@dataclass(eq=False)
class Block:
    """A block of ops with arguments, parameters, and captures.

    Ops are derived by walking the use-def graph from the result value.

    Parameters and arguments have semantics defined by the op holding the block.
    - ``parameters`` are :class:`BlockParameter` values
    - ``args`` are :class:`BlockArgument` values

    ``captures`` are outer-scope values referenced directly; they are leaves
    in block.ops (the walk stops at capture boundaries) but are always explicitly
    present to locally analyze block dependencies.
    """

    result: dgen.Value
    args: list[BlockArgument] = field(default_factory=list)
    parameters: list[BlockParameter] = field(default_factory=list)
    captures: list[dgen.Value] = field(default_factory=list)

    @property
    def dependencies(self) -> Iterator[dgen.Value]:
        """Outer-scope Value dependencies: captures and argument/parameter types."""
        yield from self.captures
        for param in self.parameters:
            yield param.type
        for arg in self.args:
            yield arg.type

    @property
    def ops(self) -> list[dgen.Op]:
        return [
            v
            for v in transitive_dependencies(self.result, stop=self.captures)
            if isinstance(v, dgen.Op)
        ]

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            yield from op.asm
