from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import dgen

from dgen.ir.traversal import transitive_dependencies
from .type import Fields, Memory, TypeType, Value


class _InstanceField:
    """Data descriptor that stores/retrieves from instance __dict__.

    Installed on Block to shadow the inherited Value.parameters property
    so that Block.parameters behaves as a plain instance attribute.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __get__(self, obj: object, objtype: type | None = None) -> object:
        if obj is None:
            return self
        return obj.__dict__[self._name]

    def __set__(self, obj: object, value: object) -> None:
        obj.__dict__[self._name] = value


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
class Block(Value):
    """A block of ops with arguments, parameters, and captures.

    Block is a Value whose type is ``builtin.Block<param_types, arg_types,
    result_type>``.  Ops reference blocks as ordinary operands.

    Ops are derived by walking the use-def graph from the result value.

    ``captures`` are outer-scope values referenced directly; they are leaves
    in block.ops (the walk stops at capture boundaries) but are always explicitly
    present to locally analyze block dependencies.
    """

    __operands__: ClassVar[Fields] = ()
    __params__: ClassVar[Fields] = ()
    __blocks__: ClassVar[tuple[str, ...]] = ()

    result: dgen.Value
    args: list[BlockArgument] = field(default_factory=list)
    parameters: list[BlockParameter] = field(default_factory=list)
    captures: list[dgen.Value] = field(default_factory=list)
    name: str | None = None

    @cached_property
    def type(self) -> Value[TypeType]:
        from dgen.builtins import pack
        from dgen.dialects.builtin import Block as BlockType

        return BlockType(
            block_parameters=pack(p.type for p in self.parameters),
            block_arguments=pack(a.type for a in self.args),
            result_type=self.result.type,
        )

    @property
    def dependencies(self) -> Iterator[dgen.Value]:
        """The block type (encodes param/arg/result types) plus captures."""
        yield self.type
        yield from self.captures

    @property
    def compile_dependencies(self) -> Iterator[dgen.Value]:
        yield self.type

    @property
    def ready(self) -> bool:
        return all(a.type.ready for a in self.args) and all(
            p.type.ready for p in self.parameters
        )

    @property
    def __constant__(self) -> Memory:
        raise TypeError("Block is not a constant")

    @property
    def values(self) -> Iterator[dgen.Value]:
        return transitive_dependencies(self.result, stop=self.captures)

    @property
    def ops(self) -> Iterator[dgen.Op]:
        return (v for v in self.values if isinstance(v, dgen.Op))

    def replace_uses_of(self, old: dgen.Value, new: dgen.Value) -> None:
        """Replace all references to old with new in block values and metadata."""
        # Sweep block.values FIRST — captures define the stop set for
        # transitive_dependencies, so they must still contain the old value
        # during this walk so ops that reference it are found.
        for v in self.values:
            v.replace_uses_of(old, new)
        # Then update block metadata
        self.captures = [new if c is old else c for c in self.captures]
        for arg in self.args:
            if arg.type is old:
                arg.type = new
        for param in self.parameters:
            if param.type is old:
                param.type = new
        if self.result is old:
            self.result = new
        # Invalidate cached type since arg/param types or result may have changed
        self.__dict__.pop("type", None)


# Shadow Value.parameters property so Block.parameters works as a plain field.
Block.parameters = _InstanceField("parameters")  # type: ignore[assignment]
