from __future__ import annotations

import ctypes as _ctypes
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

    __slots__ = ("type", "buffer", "_ct_buf")

    def __init__(self, type: Type, buffer: bytearray | None = None) -> None:
        self.type = type
        layout = type.__layout__
        self.buffer = bytearray(layout.struct.size) if buffer is None else buffer
        self._ct_buf = (_ctypes.c_char * len(self.buffer)).from_buffer(self.buffer)

    @property
    def layout(self) -> Layout:
        return self.type.__layout__

    def pack(self, *values: object) -> None:
        self.layout.struct.pack_into(self.buffer, 0, *values)

    def unpack(self) -> tuple[Any, ...]:
        return self.layout.struct.unpack(self.buffer)

    @classmethod
    def from_value(cls, type: _HasLayout, value: object) -> Memory:
        """Create Memory from a Type and a Python value."""
        parsed = type.__layout__.parse(value)
        mem = cls(type)
        if isinstance(parsed, list):
            mem.pack(*parsed)
        else:
            mem.pack(parsed)
        return mem

    @classmethod
    def from_asm(cls, type: _HasLayout, text: str) -> Memory:
        """Create Memory from a Type and an ASM literal string."""
        from dgen.asm.parser import IRParser, parse_expr

        parser = IRParser(text)
        value = parse_expr(parser)
        return cls.from_value(type, value)

    @property
    def ptr(self) -> _ctypes.c_void_p:
        """ctypes void pointer to the buffer."""
        return _ctypes.cast(self._ct_buf, _ctypes.c_void_p)

    @property
    def address(self) -> int:
        """Raw memory address of the buffer."""
        return _ctypes.addressof(self._ct_buf)
