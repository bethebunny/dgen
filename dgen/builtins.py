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
    Nil,
    Span,
    Tuple,
    builtin,
)
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


def pack(values: Iterable[Value] = ()) -> PackOp:
    """Create a PackOp with a type that honestly describes its contents.

    - Empty: ``Span<Nil>`` (no elements to consult).
    - Homogeneous (all elements share a type): ``Span<T>``.
    - Heterogeneous: ``Tuple<types>`` where ``types`` is itself a homogeneous
      ``Span<TypeType>`` PackOp listing each element's type.

    The previous heuristic always produced ``Span<first.type>``, which silently
    lied about heterogeneous mixes (a String following an Index would be tagged
    ``Span<Index>``). That lie crashes ``__constant__`` and would mis-serialize
    if anyone read ``pack_op.type``. The Tuple/Record layout is the honest
    runtime representation for a fixed-N positional bundle of mixed types.
    """
    vals = list(values)
    return PackOp(values=vals, type=_pack_type([v.type for v in vals]))


def _pack_type(element_types: list[Value[TypeType]]) -> Type:
    """Pick the honest Span/Tuple type for a PackOp from its element types."""
    if not element_types:
        return Span(pointee=Nil())
    first = element_types[0]
    if all(_same_type(t, first) for t in element_types[1:]):
        return Span(pointee=first)
    types_pack = PackOp(values=list(element_types), type=Span(pointee=TypeType()))
    return Tuple(types=types_pack)


def _same_type(a: Value[TypeType], b: Value[TypeType]) -> bool:
    """Identity-or-structural equality on a pair of element type values.

    ``a is b`` is the cheap path (most BlockArguments share a Type instance).
    For separately-constructed types we fall back to structural equivalence so
    e.g. two ``Index()`` instances still count as homogeneous.
    """
    if a is b:
        return True
    if isinstance(a, Type) and isinstance(b, Type):
        return types_equivalent(a, b)
    return False


def unpack(val: Value) -> list[Value]:
    """Unpack a PackOp into its elements, or wrap a single value in a list."""
    return list(val) if isinstance(val, PackOp) else [val]


# ===----------------------------------------------------------------------=== #
# Monkey-patches (activated on import)
# ===----------------------------------------------------------------------=== #

builtin.type("Type")(TypeType)
