from __future__ import annotations

import ctypes
from copy import deepcopy as _deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Generic, Iterator, Self, TypeVar

import dgen

from .layout import Layout, Record, _bytearray_address
from .layout import String as StringLayout

T = TypeVar("T", bound="Type")


class Value(Generic[T]):
    """Base class for SSA values."""

    name: str | None = None
    type: T

    def __init__(self, *, name: str | None = None, type: T) -> None:
        self.name = name
        self.type = type

    @property
    def ready(self) -> bool:
        return False

    @property
    def operands(self) -> list[Value]:
        return []

    @property
    def blocks(self) -> dict[str, dgen.Block]:
        return {}

    @property
    def __constant__(self) -> Memory[T]:
        raise NotImplementedError


class Type(Value["TypeType"]):
    """Any dialect type.

    Types registered via @dialect.type() get _asm_name and dialect set
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

    # Type subclasses are @dataclass(frozen=True) with their own __init__.
    # This prevents falling through to Value.__init__ for bare Type().
    def __init__(self) -> None:
        pass

    def constant(self, value: object) -> Constant[Self]:
        """Create a Constant wrapping this type and a Python value."""
        return Constant(type=self, value=Memory.from_value(self, value))

    @cached_property
    def type(self) -> TypeType:
        return TypeType(concrete=self)

    @property
    def ready(self) -> bool:
        return all(val.ready for _, val in self.parameters)

    @cached_property
    def __constant__(self) -> Memory[TypeType]:
        tt = self.type
        data: dict[str, object] = {"tag": self._asm_tag}
        for name, val in self.parameters:
            data[name] = val.__constant__.to_json()
        return Memory.from_json(tt, data)

    @cached_property
    def _asm_tag(self) -> str:
        cls = type(self)
        dialect = getattr(cls, "dialect", None)
        prefix = (
            f"{dialect.name}."
            if dialect is not None and dialect.name != "builtin"
            else ""
        )
        return f"{prefix}{getattr(cls, '_asm_name', type(self).__name__)}"

    @property
    def type_layout(self) -> Record:
        """Layout for this type as a value (tag + params)."""
        fields: list[tuple[str, Layout]] = [("tag", StringLayout())]
        for name, val in self.parameters:
            if isinstance(val, Type):
                fields.append((name, val.type_layout))
            else:
                fields.append((name, val.__constant__.type.__layout__))
        return Record(fields)

    @property
    def parameters(self) -> Iterator[tuple[str, Value]]:
        for name, field in self.__params__:
            yield name, getattr(self, name)


@dataclass(eq=False, kw_only=True)
class Constant(Value[T]):
    type: T
    value: Memory[T]

    @property
    def ready(self) -> bool:
        return True

    @property
    def __constant__(self) -> Memory[T]:
        return self.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constant):
            return NotImplemented
        if self.type != other.type:
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash((type(self), self.type, self.value))


Field = tuple[str, type[Type]]
Fields = tuple[Field, ...]


@dataclass(frozen=True)
class TypeType(Type):
    """A type whose values are themselves types.

    TypeType(concrete=Index()) wraps Index as a first-class value.
    Its __layout__ delegates to the concrete type's type_layout.
    """

    concrete: Type
    __params__: ClassVar[Fields] = (("concrete", Type),)

    @property
    def __layout__(self) -> Layout:
        return self.concrete.type_layout

    @cached_property
    def type(self) -> TypeType:
        return self


class Memory(Generic[T]):
    """Typed memory buffer — the ABI for a type.

    For pointer-based layouts (FatPointer, Pointer), the buffer contains
    raw pointers into backing data stored in `origins`. Origins are shared
    on deepcopy (immutable constant data) so packed pointers stay valid.
    """

    type: T
    buffer: bytearray
    origins: list[bytearray]

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
        from dgen.asm.parser import IRParser, parse_expr

        parser = IRParser(text)
        value = parse_expr(parser)
        return cls.from_value(type, value)

    def __deepcopy__(self, memo: dict) -> Memory[T]:
        """Copy buffer, share origins (immutable constant data)."""
        new: Memory[T] = Memory.__new__(Memory)
        memo[id(self)] = new
        new.type = _deepcopy(self.type, memo)
        new.buffer = bytearray(self.buffer)
        new.origins = self.origins  # share, not copy
        return new

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Memory):
            return NotImplemented
        return self.type == other.type and self.buffer == other.buffer

    def __hash__(self) -> int:
        return hash((self.type, bytes(self.buffer)))

    def __repr__(self) -> str:
        return f"Memory({self.type!r}, {self.unpack()!r})"

    @property
    def address(self) -> int:
        """Raw memory address of the buffer."""
        return _bytearray_address(self.buffer)
