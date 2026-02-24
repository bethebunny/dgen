from __future__ import annotations

import ctypes
from typing import Any, Protocol

from .layout import Layout


class Type(Protocol):
    """Any dialect type.

    Types registered via @dialect.type() get _asm_name and dialect set
    automatically. The format_expr() function handles formatting via
    type_asm() for registered types, or falls back to .asm for types
    with hand-written formatting (e.g. llvm types).
    """

    __layout__: Layout

    ...


class Memory:
    """Typed memory buffer — the ABI for a type."""

    type: Type
    buffer: bytearray

    def __init__(self, type: Type, buffer: bytearray | None = None) -> None:
        self.type = type
        self.buffer = bytearray(self.layout.byte_size) if buffer is None else buffer

    @property
    def layout(self) -> Layout:
        return self.type.__layout__

    def pack(self, *values: object) -> None:
        self.layout.struct.pack_into(self.buffer, 0, *values)

    def unpack(self) -> tuple[Any, ...]:
        return self.layout.struct.unpack(self.buffer)

    @classmethod
    def from_value(cls, type: Type, value: object) -> Memory:
        """Create Memory from a Type and a Python value."""
        parsed = type.__layout__.parse(value)
        mem = cls(type)
        if isinstance(parsed, list):
            mem.pack(*parsed)
        else:
            mem.pack(parsed)
        return mem

    @classmethod
    def from_asm(cls, type: Type, text: str) -> Memory:
        """Create Memory from a Type and an ASM literal string."""
        from dgen.asm.parser import IRParser, parse_expr

        parser = IRParser(text)
        value = parse_expr(parser)
        return cls.from_value(type, value)

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
        ct = (ctypes.c_char * len(self.buffer)).from_buffer(self.buffer)
        return ctypes.addressof(ct)
