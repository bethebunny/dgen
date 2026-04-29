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
    format_value,
    constant,
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
    """Sugar op: wraps multiple values into a single list-like operand.

    Never emitted standalone in ASM — the formatter inlines it as [...].
    The codegen skips it entirely.
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

    When every element is itself a ``Constant``, the result folds to a
    plain ``Constant`` of the appropriate aggregate type — at codegen, no
    PackOp survives in compile-time positions because there's nothing to
    fold (or the operands themselves were unfoldable runtime values).

    When any element is non-constant (a runtime SSA value, a ``Type``
    instance — Types are *not* constants because their parameters can
    depend on runtime values, etc.), the result is a ``PackOp`` carrying
    those values for later emission as an ``insertvalue`` chain.
    """
    vals = list(values)
    ty = _pack_type([v.type for v in vals])
    if all(isinstance(v, Constant) for v in vals):
        json_list = [v.value.to_json() for v in vals]
        return Constant(type=ty, value=Memory.from_json(ty, json_list))
    return PackOp(values=vals, type=ty)


def _pack_type(element_types: list[Value[TypeType]]) -> Type:
    """Pick the honest Array/Tuple type for a pack from its element types."""
    n = Index().constant(len(element_types))
    if not element_types:
        return Array(element_type=Nil(), n=n)
    first = element_types[0]
    if all(_same_type(t, first) for t in element_types[1:]):
        return Array(element_type=first, n=n)
    # Heterogeneous: wrap each element type as ``Constant<TypeType>`` so
    # ``pack`` folds the inner types-list to a Constant aggregate. This
    # keeps the ``Tuple<types=...>`` parameter compile-time even when the
    # outer pack itself is a runtime PackOp of mixed-type values.
    types_pack = pack(TypeType().constant(constant(t)) for t in element_types)
    return Tuple(types=types_pack)


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
    ``Constant`` (``Array`` / ``Tuple`` typed) gets reconstructed: each
    element is wrapped back as a ``Constant`` of its field type, or
    yielded directly when the rich form is already a ``Value`` (i.e. a
    ``Type`` instance for ``TypeType``-typed fields).

    Anything else (a runtime SSA value with no aggregate structure)
    wraps in a singleton list.
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
    # Tuple<types>: each field's type comes from ``types``, rich form is a
    # dict keyed by stringified positional index.
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


def _format_aggregate_constant(c: Constant, slot: SlotFn) -> str:
    """Format an aggregate ``Constant`` as inline ``[elem0, elem1, ...]``.

    Each element is materialised via ``_aggregate_elements`` and formatted
    by its own ``format_asm`` — a Type instance prints as its name, a
    scalar Constant prints as ``Type(value)``, and a nested aggregate
    recurses into this same path.
    """
    return "[" + ", ".join(format_value(e, slot) for e in _aggregate_elements(c)) + "]"


# Override Constant.format_asm so aggregate-typed constants format as
# inline lists rather than the verbose ``Tuple<...>(dict)`` shape that
# the default ``Type(value)`` formatter would produce. Keeps the surface
# syntax for compile-time aggregates the same as the inline ``[a, b]``
# sugar from ``PackOp``.
def _constant_format_asm(self: Constant, slot: SlotFn = _default_slot) -> str:
    if isinstance(self.type, (Array, Tuple)):
        return _format_aggregate_constant(self, slot)
    body = format_value(constant(self), slot)
    return f"{self.type.format_asm(slot)}({body})"


Constant.format_asm = _constant_format_asm


# ===----------------------------------------------------------------------=== #
# Monkey-patches (activated on import)
# ===----------------------------------------------------------------------=== #

builtin.type("Type")(TypeType)
