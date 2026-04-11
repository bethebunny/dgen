from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Generic, Iterator, Self, TypeVar

import dgen

from .dialect import Dialect
from .layout import Layout, TypeValue
from .memory import Memory

T = TypeVar("T", bound="Type")

# Slot function: maps a Value to its SSA name string (without the % prefix).
SlotFn = Callable[["Value"], str]


def _default_slot(v: Value) -> str:
    if v.name is None:
        raise ValueError(f"Unnamed value {v!r} requires a SlotTracker")
    return v.name


class Value(Generic[T]):
    """Base class for SSA values."""

    __params__: ClassVar[Fields] = ()
    __operands__: ClassVar[Fields] = ()
    __blocks__: ClassVar[tuple[str, ...]] = ()
    __constraints__: ClassVar[tuple[object, ...]] = ()
    name: str | None = None
    type: Value[TypeType]

    def __init__(self, *, name: str | None = None, type: T) -> None:
        self.name = name
        self.type = type

    @property
    def parameters(self) -> Iterator[tuple[str, Value[TypeType]]]:
        for name, field in self.__params__:
            yield name, getattr(self, name)

    @property
    def operands(self) -> Iterator[tuple[str, Value]]:
        for name, _ in self.__operands__:
            yield name, getattr(self, name)

    @property
    def blocks(self) -> Iterator[tuple[str, dgen.Block]]:
        for name in self.__blocks__:
            yield name, getattr(self, name)

    @property
    def dependencies(self) -> Iterator[Value]:
        """All Value dependencies for use-def graph traversal."""
        yield self.type
        for _, param in self.parameters:
            yield param
        for _, operand in self.operands:
            yield operand
        for _, block in self.blocks:
            yield from block.dependencies

    @property
    def compile_dependencies(self) -> Iterator[Value]:
        """Compile-time dependencies — values that must be materialized
        before this value can be lowered.

        Yields the type, the compile-time parameters, and the arg/parameter
        types of any owned blocks. Excludes operands (runtime dataflow) and
        block captures (runtime references from inner scopes to outer ones).
        """
        yield self.type
        for _, param in self.parameters:
            yield param
        for _, block in self.blocks:
            for arg in block.args:
                yield arg.type
            for block_param in block.parameters:
                yield block_param.type

    @property
    def __constant__(self) -> Memory[T]:
        raise NotImplementedError

    def required_dialects(self) -> Iterator[Dialect]:
        """Dialects that must be importable for this value to round-trip
        through ASM. The default is empty — overridden by ``Constant`` to
        walk the rich payload and yield dialects of any embedded ``Type``
        instances that aren't reachable via the use-def graph.
        """
        return iter(())

    @property
    def ready(self) -> bool:
        return self.type.ready and all(val.ready for _, val in self.parameters)

    def format_asm(self, slot: SlotFn = _default_slot) -> str:
        """Format as ASM expression. Default: SSA reference ``%name``."""
        return f"%{slot(self)}"

    def has_trait(self, trait: type) -> bool:
        """Check whether this value implements a trait."""
        return isinstance(self, trait)

    def replace_operand(self, old: Value, new: Value) -> None:
        """Replace all occurrences of old with new in operand fields."""
        for name, _ in self.__operands__:
            if getattr(self, name) is old:
                setattr(self, name, new)

    def replace_uses_of(self, old: Value, new: Value) -> None:
        """Replace all references to old with new in this value's fields and owned blocks."""
        self.replace_operand(old, new)
        for name, val in self.parameters:
            if val is old:
                setattr(self, name, new)
        if self.type is old:
            self.type = new
        for _, block in self.blocks:
            block.replace_uses_of(old, new)


def constant(value: Value) -> object:
    """Resolve a constant ``Value`` to its rich Python representation.

    For a ``Type`` instance, returns the type itself (TypeType is the
    identity case). For any other ``Value`` whose payload can be
    materialised, returns the rich form: scalars for primitive types,
    lists for spans/arrays, dicts for records, and ``Type`` instances at
    every type-valued leaf.

    Subsumes the previous ``type_constant`` (TypeType ↦ Type) and the
    ``v.__constant__.to_native_value()`` pattern with a single free function
    that reads cleanly at call sites.
    """
    if isinstance(value, Type):
        return value
    return value.__constant__.to_native_value()


class Type(Value["TypeType"]):
    """Any dialect type.

    Types registered via @dialect.type() get asm_name and dialect set
    automatically. The format_expr() function handles formatting via
    type_asm() for registered types, or falls back to .asm for types
    with hand-written formatting (e.g. llvm types).

    Type subclasses are typically @dataclass(frozen=True) and provide
    their own __init__. This class does not call Value.__init__ since
    .type and .name are provided via cached_property/class default.
    """

    __layout__: Layout
    __params__: ClassVar[Fields] = ()
    name: None = None
    dialect: dgen.Dialect
    asm_name: str

    # Type subclasses are @dataclass(frozen=True) with their own __init__.
    # This prevents falling through to Value.__init__ for bare Type().
    def __init__(self) -> None:
        pass

    def constant(self, value: object) -> Constant[Self]:
        """Create a Constant wrapping this type and a Python value."""
        return Constant(type=self, value=Memory.from_value(self, value))

    @cached_property
    def type(self) -> TypeType:
        return TypeType()

    def to_json(self) -> dict[str, object]:
        """Serialize to a self-describing dict.

        Format::

            {"tag": "dialect.Name", "params": {"p": {"type": ..., "value": ...}, ...}}

        Each param carries its own type descriptor so deserialization needs no schema.
        """
        return {
            "tag": self.qualified_name,
            "params": {
                name: {
                    "type": constant(param.type).to_json(),
                    "value": param.__constant__.to_json(),
                }
                for name, param in self.parameters
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> Type:
        """Reconstruct a Type from a self-describing dict."""
        tag = data["tag"]
        assert isinstance(tag, str)
        dialect_name, type_name = tag.split(".")
        type_cls = Dialect.get(dialect_name).types[type_name]
        return type_cls(
            **{
                name: Type.from_json(tv["type"]).constant(tv["value"])
                for name, tv in data["params"].items()
            }
        )

    @cached_property
    def __constant__(self) -> Memory[TypeType]:
        return Memory.from_json(TypeType(), self.to_json())

    def format_asm(self, slot: SlotFn = _default_slot) -> str:
        """Format as ``dialect.Name<params>`` (no prefix for builtin)."""
        name = self.dialect.qualified_name(self.asm_name)
        params = list(self.parameters)
        if not params:
            return name
        args = ", ".join(format_value(val, slot) for _, val in params)
        return f"{name}<{args}>"

    @cached_property
    def qualified_name(self) -> str:
        return f"{self.dialect.name}.{self.asm_name}"

    @property
    def parameters(self) -> Iterator[tuple[str, Value]]:
        for name, field in self.__params__:
            yield name, getattr(self, name)

    @property
    def dependencies(self) -> Iterator[Value]:
        """Types only depend on their parameters — skip type/operands/blocks."""
        for _, param in self.parameters:
            yield param

    @property
    def compile_dependencies(self) -> Iterator[Value]:
        """Same as ``dependencies`` for Types — params only."""
        for _, param in self.parameters:
            yield param


@dataclass(eq=False, kw_only=True)
class Constant(Value[T]):
    type: T
    value: Memory[T]

    def format_asm(self, slot: SlotFn = _default_slot) -> str:
        """Format as Type(value) — always includes the type prefix."""
        body = format_value(constant(self), slot)
        return f"{self.type.format_asm(slot)}({body})"

    @property
    def ready(self) -> bool:
        return True

    @property
    def __constant__(self) -> Memory[T]:
        return self.value

    def required_dialects(self) -> Iterator[Dialect]:
        """Walk the rich payload and yield dialects of any embedded
        ``Type`` instances. Constants can carry type references in their
        bytes that aren't reachable via ``Value.dependencies`` (e.g. a
        TypeType constant whose value is from a non-builtin dialect, or
        an existential whose witness type is). The asm formatter calls
        this when collecting imports.
        """
        return _walk_for_dialects(constant(self))


def _walk_for_dialects(value: object) -> Iterator[Dialect]:
    if isinstance(value, Type):
        yield value.dialect
        for _, param in value.parameters:
            yield from _walk_for_dialects(param)
    elif isinstance(value, Value):
        yield from _walk_for_dialects(value.type)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_for_dialects(v)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_for_dialects(item)


Fields = tuple[tuple[str, type[Type]], ...]


class TypeType(Type):
    """A type whose values are themselves types.

    TypeType() is the metatype — its values are type descriptors.
    The concrete identity of a type value is encoded in the TypeValue
    layout (self-describing via tag), not in the TypeType itself.
    """

    __layout__ = TypeValue()

    @property
    def ready(self) -> bool:
        return True

    @cached_property
    def type(self) -> TypeType:
        # TypeType is its own metatype — break the infinite recursion.
        # Subclasses (Trait and its children) use Type.type → TypeType().
        if type(self) is TypeType:
            return self
        return TypeType()


# ===----------------------------------------------------------------------=== #
# ASM formatting helpers
# ===----------------------------------------------------------------------=== #


def _format_float(v: float) -> str:
    iv = int(v)
    if float(iv) == v:
        return f"{iv}.0"
    return str(v)


def format_value(value: object, slot: SlotFn = _default_slot) -> str:
    """Format a value as an ASM expression.

    ``Value`` instances (including ``Type``) dispatch to ``format_asm`` and
    print as named SSA references / type-ASM sugar. Plain Python literals
    (int, float, str, list, dict, None) format as ASM literals.
    """
    if isinstance(value, Value):
        return value.format_asm(slot)
    if isinstance(value, list):
        return "[" + ", ".join(format_value(v, slot) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f'"{k}": {format_value(v, slot)}' for k, v in value.items())
        return "{" + items + "}"
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if value is None:
        return "()"
    raise ValueError(f"Cannot format {type(value).__name__} as ASM literal: {value!r}")
