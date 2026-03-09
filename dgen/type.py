from __future__ import annotations

import ctypes
from copy import deepcopy as _deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Generic, Iterator, Self, TypeVar

import dgen

from .layout import Layout, TypeValue, _bytearray_address

T = TypeVar("T", bound="Type")


class Value(Generic[T]):
    """Base class for SSA values."""

    __params__: ClassVar[Fields] = ()
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
    def __constant__(self) -> Memory[T]:
        raise NotImplementedError

    @property
    def ready(self) -> bool:
        return self.type.ready and all(val.ready for _, val in self.parameters)


def type_constant(value: Value[TypeType]) -> Type:
    """Resolve a Value[TypeType] to a concrete Type."""
    if isinstance(value, Type):
        return value
    data = value.__constant__.to_json()
    assert isinstance(data, dict)
    return _type_from_dict(data)


def _type_from_dict(data: dict[str, object]) -> Type:
    """Reconstruct a Type from its serialized TypeType dict."""
    from .dialect import Dialect

    tag = data["tag"]
    assert isinstance(tag, str)
    dialect_name, type_name = tag.split(".")
    dialect = Dialect.get(dialect_name)
    cls = dialect.types[type_name]
    params = {k: v for k, v in data.items() if k != "tag"}
    if not params:
        return cls()
    kwargs: dict[str, object] = {}
    for param_name, param_value in params.items():
        for field_name, field_type in cls.__params__:
            if field_name == param_name:
                if isinstance(param_value, list):
                    kwargs[param_name] = [
                        _type_from_dict(v)
                        if isinstance(v, dict)
                        else field_type().constant(v)
                        for v in param_value
                    ]
                elif isinstance(param_value, dict):
                    kwargs[param_name] = _type_from_dict(param_value)
                else:
                    kwargs[param_name] = field_type().constant(param_value)
                break
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
        return TypeType(concrete=self)

    @cached_property
    def __constant__(self) -> Memory[TypeType]:
        data: dict[str, object] = {"tag": self.qualified_name}
        for name, param in self.parameters:
            if isinstance(param, list):
                data[name] = [p.__constant__.to_json() for p in param]
            else:
                data[name] = param.__constant__.to_json()
        return Memory.from_json(self.type, data)

    @cached_property
    def qualified_name(self) -> str:
        return f"{self.dialect.name}.{self.asm_name}"

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
    """

    concrete: Value[TypeType]
    __params__: ClassVar[Fields] = (("concrete", Type),)

    @property
    def __layout__(self) -> TypeValue:
        """Layout for this type as a value — a pointer to a self-describing Record."""
        return TypeValue()

    @property
    def ready(self) -> bool:
        return isinstance(self.concrete, Type) or self.concrete.ready

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
