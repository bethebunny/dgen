from __future__ import annotations

import ctypes
from copy import deepcopy as _deepcopy
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterator, Self, TypeVar

from .layout import Layout, _bytearray_address

if TYPE_CHECKING:
    from .value import Constant


class Type:
    """Any dialect type.

    Types registered via @dialect.type() get _asm_name and dialect set
    automatically. The format_expr() function handles formatting via
    type_asm() for registered types, or falls back to .asm for types
    with hand-written formatting (e.g. llvm types).
    """

    __layout__: ClassVar[Layout]
    __params__: ClassVar[Fields] = ()

    def constant(self, value: object) -> Constant[Self]:
        """Create a Constant wrapping this type and a Python value."""
        from .value import Constant

        return Constant(type=self, value=Memory.from_value(self, value))

    @classmethod
    def for_value(cls, value: object) -> Type:
        return cls()

    @property
    def parameters(self) -> Iterator[tuple[str, Type]]:
        for name, field in self.__params__:
            yield name, getattr(self, name)


Field = tuple[str, type[Type]]
Fields = tuple[Field, ...]


T = TypeVar("T", bound=Type)


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

    def pack(self, *values: object) -> None:
        self.layout.struct.pack_into(self.buffer, 0, *values)

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
        result = self.layout.to_json(self.buffer, 0)
        hook = getattr(self.type, "__to_json__", None)
        if hook is not None:
            result = hook(result)
        return result

    @classmethod
    def from_json(cls, type: Type, value: object) -> Memory:
        """Create Memory from a Type and a JSON-compatible Python value."""
        hook = getattr(type, "__from_json__", None)
        if hook is not None:
            value = hook(value)
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
