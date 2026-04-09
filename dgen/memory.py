"""Typed memory buffer — the ABI-level representation of values."""

from __future__ import annotations

import ctypes
from copy import deepcopy as _deepcopy
from typing import Any, Generic, TypeVar

from .layout import Layout, _bytearray_address

T = TypeVar("T", bound="Type")

# Avoid circular import: type.py imports memory.py, not the reverse.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .type import Type


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
        return (
            self.type.__constant__.to_json() == other.type.__constant__.to_json()
            and self.to_json() == other.to_json()
        )

    def __hash__(self) -> int:
        return hash((bytes(self.type.__constant__.buffer), bytes(self.buffer)))

    def __repr__(self) -> str:
        return f"Memory({self.type!r}, {self.unpack()!r})"

    @property
    def address(self) -> int:
        """Raw memory address of the buffer."""
        return _bytearray_address(self.buffer)
