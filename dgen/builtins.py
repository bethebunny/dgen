"""Hand-written builtin ops: ConstantOp, PackOp, and helpers.

These complement the generated builtin dialect but live outside the dialect
file to keep it purely generated.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar

from dgen import Constant, Op, Type, TypeType, Value
from dgen.dialects.builtin import (
    Array,
    Nil,
    Tuple,
    builtin,
)
from dgen.dialects.index import Index
from dgen.memory import Memory
from dgen.type import (
    Fields,
    SlotFn,
    _default_slot,
    constant,
    format_value,
    is_constant,
    types_equivalent,
)


# ===----------------------------------------------------------------------=== #
# ConstantOp (custom __init__, multiple inheritance)
# ===----------------------------------------------------------------------=== #


@builtin.op("constant")
@dataclass(eq=False, kw_only=True)
class ConstantOp(Op, Constant):
    value: Memory | object
    type: Value[TypeType]

    format_asm = Value.format_asm  # SSA reference, not Constant literal

    @property
    def __constant__(self) -> Memory:
        if isinstance(self.value, Memory):
            return self.value
        # Deferred: type was an SSA ref at parse time, resolve now.
        return Memory.from_value(constant(self.type), self.value)

    @classmethod
    def from_constant(
        cls, constant: Constant, *, name: str | None = None
    ) -> ConstantOp:
        return cls(value=constant.value, type=constant.type, name=name)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


# ===----------------------------------------------------------------------=== #
# PackOp (sugar op, never emitted standalone in ASM)
# ===----------------------------------------------------------------------=== #


@builtin.op("pack")
@dataclass(eq=False, kw_only=True)
class PackOp(Op):
    """Sugar op: wraps multiple runtime values into a single tuple.

    Only exists when ``pack()`` couldn't fold its input to a ``Constant``
    — i.e. some element is a runtime SSA value. Emitted at codegen as an
    ``insertvalue`` chain producing the bundle aggregate.
    """

    values: list[Value]
    type: Type
    __operands__: ClassVar[Fields] = ()

    def format_asm(self, slot: SlotFn = _default_slot) -> str:
        """PackOp always inlines as ``[elem, elem, ...]`` sugar."""
        return "[" + ", ".join(format_value(v, slot) for v in self.values) + "]"

    def __iter__(self) -> Iterator[Value]:
        return iter(self.values)

    @property
    def operands(self) -> Iterator[tuple[str, Value]]:
        for i, v in enumerate(self.values):
            yield f"values[{i}]", v

    def replace_operand(self, old: Value, new: Value) -> None:
        self.values = [new if v is old else v for v in self.values]

    @property
    def __constant__(self) -> Memory:
        json_list = [v.__constant__.to_json() for v in self.values]
        return Memory.from_json(self.type, json_list)


def pack(values: Iterable[Value] = ()) -> PackOp | Constant:
    """Build the value bundle for a list of values.

    Folds to a plain ``Constant`` of the appropriate aggregate type when
    every element is itself a constant in the dgen sense (``is_constant``
    — i.e. ``constant(v)`` succeeds: the value materialises to a Python
    rich form, no runtime SSA dependencies). Otherwise produces a
    ``PackOp`` carrying the runtime values for codegen.
    """
    vals = list(values)
    ty = _pack_type([v.type for v in vals])
    if all(is_constant(v) for v in vals):
        return ty.constant([constant(v) for v in vals])
    return PackOp(values=vals, type=ty)


def _pack_type(element_types: list[Value[TypeType]]) -> Type:
    """Pick the Array/Tuple type that honestly describes a pack's contents."""
    n = Index().constant(len(element_types))
    if not element_types:
        return Array(element_type=Nil(), n=n)
    first = element_types[0]
    if all(_same_type(t, first) for t in element_types[1:]):
        return Array(element_type=first, n=n)
    return Tuple(types=pack(element_types))


def _same_type(a: Value[TypeType], b: Value[TypeType]) -> bool:
    """Identity-or-structural equality on a pair of element type values."""
    if a is b:
        return True
    if isinstance(a, Type) and isinstance(b, Type):
        return types_equivalent(a, b)
    return False


def unpack(val: Value) -> list[Value]:
    """Decompose a tuple-shaped value into its element Values.

    A ``PackOp`` yields its ``.values`` directly. An aggregate
    ``Constant`` (``Array``/``Tuple``-typed) gets reconstructed:
    each element is wrapped back as a ``Constant`` of its field type,
    or yielded directly when the rich form is already a ``Value`` (a
    ``Type`` instance for ``TypeType``-typed fields).
    """
    if isinstance(val, PackOp):
        return list(val)
    if isinstance(val, Constant) and isinstance(val.type, (Array, Tuple)):
        return list(_aggregate_elements(val))
    return [val]


def _aggregate_elements(c: Constant) -> Iterator[Value]:
    """Yield each element of an aggregate Constant, wrapped as a Value of
    its field type (or yielded directly when the rich form is already a
    Value, e.g. a Type instance for TypeType-typed fields)."""
    rich = c.value.to_native_value()
    if isinstance(c.type, Array):
        elem_type = constant(c.type.element_type)
        assert isinstance(elem_type, Type)
        for v in rich:
            yield _wrap_field(elem_type, v)
        return
    elem_types = constant(c.type.types)
    assert isinstance(elem_types, list)
    for i, t in enumerate(elem_types):
        yield _wrap_field(t, rich[str(i)])


def _wrap_field(field_type: Type, rich_value: object) -> Value:
    """Re-wrap a rich-form aggregate field value as a Value. Type
    instances pass through; scalars get wrapped as ``Constant``s of the
    declared field type."""
    if isinstance(rich_value, Value):
        return rich_value
    return field_type.constant(rich_value)


# ===----------------------------------------------------------------------=== #
# Monkey-patches (activated on import)
# ===----------------------------------------------------------------------=== #

builtin.type("Type")(TypeType)
