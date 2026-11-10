from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterator
from copy import deepcopy as _deepcopy
from dataclasses import dataclass
from functools import cached_property
from struct import Struct
from typing import Any, ClassVar, Generic, Self, TypeVar

import dgen

from .dialect import Dialect
from .layout import (
    STRING,
    TYPE_RECORD,
    Image,
    Layout,
    TypeValue,
    _aligned,
    _bytearray_address,
    _read,
    param_type_values,
    parse,
    type_record,
    type_record_parts,
)

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
    def __constant__(self) -> Memory[T]:
        raise NotImplementedError

    @property
    def ready(self) -> bool:
        return self.type.ready and all(val.ready for _, val in self.parameters)

    def format_asm(self, slot: SlotFn = _default_slot) -> str:
        """Format as ASM expression. Default: SSA reference ``%name``."""
        return f"%{slot(self)}"

    def has_trait(self, trait: type[Trait]) -> bool:
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


def type_constant(value: Value[TypeType]) -> Type:
    """Resolve a Value[TypeType] to a concrete Type."""
    if isinstance(value, Type):
        return value
    return _type_from_memory(value.__constant__)


def _type_from_memory(value: Memory[TypeType]) -> Type:
    """Rebuild a Type from its type value's bytes, through its record and not JSON."""
    (address,) = value.unpack()
    tag, params_record, params = type_record_parts(address)
    dialect_name, type_name = tag.split(".")
    cls = Dialect.get(dialect_name).types[type_name]
    kwargs: dict[str, Value] = {}
    for (name, lay), at, type_bytes in zip(
        params_record.fields, params_record.offsets, param_type_values(address)
    ):
        param_type = _type_from_memory(_held(TypeType(), type_bytes, value))
        kwargs[name] = _param_value(
            _held(param_type, params[at : at + lay.byte_size], value)
        )
    return cls(**kwargs)


def _param_value(memory: Memory) -> Value:
    """A type parameter's value, from its Memory: a type, a pack of values or a constant."""
    from dgen.dialects import builtin
    from dgen.module import pack

    match memory.type:
        case TypeType():
            return _type_from_memory(memory)
        case builtin.Span(pointee=pointee):
            element = type_constant(pointee)
            size = element.__layout__.byte_size
            address, count = memory.unpack()
            data = _read(address, size * count)
            return pack(
                _param_value(_held(element, data[i * size : (i + 1) * size], memory))
                for i in range(count)
            )
        case _:
            return Constant(type=memory.type, value=memory)


def _held(type: T, data: bytes | bytearray, owner: object) -> Memory[T]:
    """A Memory of type holding a copy of data, whose addresses point into owner."""
    result = Memory(type, bytearray(data))
    result.origins.append(owner)
    return result


def _type_from_dict(data: dict[str, object]) -> Type:
    """Reconstruct a Type from its serialized TypeType dict."""
    from dgen.module import pack

    tag = data["tag"]
    assert isinstance(tag, str)
    dialect_name, type_name = tag.split(".")
    dialect = Dialect.get(dialect_name)
    cls = dialect.types[type_name]
    param_types = dict(cls.__params__)
    kwargs: dict[str, object] = {}
    for param_name, param_value in data.items():
        if param_name == "tag":
            continue
        field_type = param_types[param_name]
        if isinstance(param_value, list):
            kwargs[param_name] = pack(
                _type_from_dict(v) if isinstance(v, dict) else field_type().constant(v)
                for v in param_value
            )
        elif isinstance(param_value, dict):
            kwargs[param_name] = _type_from_dict(param_value)
        else:
            kwargs[param_name] = field_type().constant(param_value)
    return cls(**kwargs)


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

    @cached_property
    def __constant__(self) -> Memory[TypeType]:
        params = [
            (name, param.__constant__, param.type.__constant__)
            for name, param in self.parameters
        ]
        result = Memory(self.type)
        result.origins += [
            memory for _, value, type in params for memory in (value, type)
        ]
        record = type_record(
            self.qualified_name,
            self.__layout__,
            [
                (name, value.layout, bytes(value.buffer), bytes(type.buffer))
                for name, value, type in params
            ],
            result.origins,
        )
        TypeValue.struct.pack_into(result.buffer, 0, record)
        return result

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


@dataclass(eq=False, kw_only=True)
class Constant(Value[T]):
    type: T
    value: Memory[T]

    def format_asm(self, slot: SlotFn = _default_slot) -> str:
        """Format as literal, with type prefix for parameterized types."""
        json_str = format_json(self.__constant__.to_json(), slot)
        if type(self.type).__params__:
            return f"{self.type.format_asm(slot)}({json_str})"
        return json_str

    @property
    def ready(self) -> bool:
        return True

    @property
    def __constant__(self) -> Memory[T]:
        return self.value


Field = tuple[str, type[Type]]
Fields = tuple[Field, ...]


class TypeType(Type):
    """A type whose values are themselves types.

    TypeType() is the metatype — its values are type descriptors.
    The concrete identity of a type value is encoded in the TypeValue
    layout (self-describing via tag), not in the TypeType itself.
    """

    @property
    def __layout__(self) -> TypeValue:
        """Layout for this type as a value — a pointer to a self-describing Record."""
        return TypeValue()

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


class Trait(TypeType):
    """Base class for all dgen traits.

    Traits are type values in the type hierarchy — they describe sets
    of types. A trait can appear wherever a type can: as a type annotation,
    in constraint checks, in ASM type position.

    """


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

    Values dispatch to ``format_asm``; plain Python literals (int, float,
    str, list, dict) are formatted as JSON.
    """
    if isinstance(value, Value):
        return value.format_asm(slot)
    return format_json(value, slot)


def format_json(value: object, slot: SlotFn = _default_slot) -> str:
    """Format a plain Python value as an ASM literal."""
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
    raise ValueError(f"Cannot format {type(value).__name__} as ASM literal: {value!r}")


# The wire header: magic, version, the type value's slot, and the offsets of
# the value and the relocation table, then the relocation count.
WIRE_HEADER = Struct("<8sQQQQQ")
WIRE_MAGIC = b"dgenwire"
WIRE_VERSION = 1
_WIRE_TYPE_AT = 16
_ADDRESS = Struct("<Q")


def _relocated(data: bytes) -> tuple[bytearray, int]:
    """A wire image with its offsets made into addresses, and the offset of its value."""
    image = bytearray(data)
    magic, version, _, value, relocations, count = WIRE_HEADER.unpack_from(image)
    if magic != WIRE_MAGIC or version != WIRE_VERSION:
        raise ValueError(f"not a dgen wire image of version {WIRE_VERSION}")
    base = _bytearray_address(image)
    for (at,) in _ADDRESS.iter_unpack(image[relocations:][: 8 * count]):
        (offset,) = _ADDRESS.unpack_from(image, at)
        _ADDRESS.pack_into(image, at, base + offset)
    return image, value


def wire_json(data: bytes) -> tuple[dict[str, object], object]:
    """The JSON of a wire image's type and value, read without the dialect registry."""
    image, value = _relocated(data)
    (record,) = _ADDRESS.unpack_from(image, _WIRE_TYPE_AT)
    layout = parse(
        STRING.to_json(_read(record, TYPE_RECORD.byte_size), TYPE_RECORD.offsets[1])
    )
    return TypeValue().to_json(image, _WIRE_TYPE_AT), layout.to_json(image, value)


class Memory(Generic[T]):
    """Typed memory buffer — the ABI for a type.

    For pointer-based layouts (Span, Pointer), the buffer contains
    raw pointers into backing data stored in `origins`. Origins also holds
    any other object whose lifetime must extend through this Memory's
    (e.g. a JIT engine for a Function pointer). Origins are shared on
    deepcopy (immutable constant data) so packed pointers stay valid.
    """

    type: T
    buffer: bytearray
    origins: list

    def __init__(self, type: T, buffer: bytearray | None = None) -> None:
        self.type = type
        self.buffer = bytearray(self.layout.byte_size) if buffer is None else buffer
        self.origins = []

    @property
    def layout(self) -> Layout:
        return self.type.__layout__

    def unpack(self) -> tuple[Any, ...]:
        return self.layout.struct.unpack(self.buffer)

    @classmethod
    def from_value(cls, type: Type, value: object) -> Memory:
        """Create Memory from a Type and a Python value.

        Converts str/bytes to list[int], then delegates to from_json().
        """
        if isinstance(value, str):
            value = value.encode("utf-8")
        if isinstance(value, bytes):
            value = list(value)
        return cls.from_json(type, value)

    def to_json(self) -> object:
        """Convert to a JSON-compatible Python value by reading from the buffer."""
        return self.layout.to_json(self.buffer, 0)

    @classmethod
    def from_json(cls, type: Type, value: object) -> Memory:
        """Create Memory from a Type and a JSON-compatible Python value."""
        mem = cls(type)
        type.__layout__.from_json(mem.buffer, 0, value, mem.origins)
        return mem

    @classmethod
    def from_raw(cls, type: Type, address: int) -> Memory:
        """Create Memory from a raw pointer address (e.g. a JIT result).

        Copies layout.byte_size bytes from the address. No origins — the
        caller's buffers must remain alive for any inner pointers.
        """
        layout = type.__layout__
        buf = bytes((ctypes.c_char * layout.byte_size).from_address(address))
        return cls(type, bytearray(buf))

    @classmethod
    def from_asm(cls, type: Type, text: str) -> Memory:
        """Create Memory from a Type and an ASM literal string."""
        from dgen.asm.parser import ASMParser, value_expression

        parser = ASMParser(text)
        value = value_expression(parser)
        return cls.from_value(type, value)

    def __deepcopy__(self, memo: dict) -> Memory[T]:
        """Copy buffer, share origins (immutable constant data)."""
        new: Memory[T] = Memory.__new__(Memory)
        memo[id(self)] = new
        new.type = _deepcopy(self.type, memo)
        new.buffer = bytearray(self.buffer)
        new.origins = self.origins  # share, not copy
        return new

    def to_wire(self) -> bytes:
        """The type and value as one position-independent image (see the wire format)."""
        image = Image(WIRE_HEADER.size)
        value = image.copy(self.address, self.layout)
        type_value = self.type.__constant__
        image.data[_WIRE_TYPE_AT : _WIRE_TYPE_AT + 8] = type_value.buffer
        type_value.layout.write(image, _WIRE_TYPE_AT)
        (type_at,) = _ADDRESS.unpack_from(image.data, _WIRE_TYPE_AT)
        relocations = _aligned(len(image.data), 8)
        image.data += bytes(relocations - len(image.data))
        image.data += b"".join(_ADDRESS.pack(at) for at in image.relocations)
        WIRE_HEADER.pack_into(
            image.data,
            0,
            WIRE_MAGIC,
            WIRE_VERSION,
            type_at,
            value,
            relocations,
            len(image.relocations),
        )
        return bytes(image.data)

    @classmethod
    def from_wire(cls, data: bytes) -> Memory:
        """The Memory that to_wire() wrote, relocated into one buffer it owns."""
        image, value = _relocated(data)
        type = _type_from_memory(_held(TypeType(), image[_WIRE_TYPE_AT:][:8], image))
        return _held(type, image[value:][: type.__layout__.byte_size], image)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Memory):
            return NotImplemented
        return self.to_wire() == other.to_wire()

    def __hash__(self) -> int:
        return hash(self.to_wire())

    def __repr__(self) -> str:
        return f"Memory({self.type!r}, {self.unpack()!r})"

    @property
    def address(self) -> int:
        """Raw memory address of the buffer."""
        return _bytearray_address(self.buffer)
