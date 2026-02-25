from __future__ import annotations

import ctypes
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Iterator,
    Self,
    TypeVar,
    get_args,
    get_origin,
)

from .layout import Layout

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

    def constant(self, value: object) -> Constant[Self]:
        """Create a Constant wrapping this type and a Python value."""
        from .value import Constant

        return Constant(type=self, value=Memory.from_value(self, value))

    @classmethod
    def for_value(cls, value: object) -> Type:
        return cls()

    @classmethod
    def fields(cls) -> Iterator[tuple[str, type[Type], bool]]:
        from .asm.formatting import _class_hints
        from .value import Constant, Value

        for name, hint in _class_hints(cls).items():
            if get_origin(hint) is Annotated:
                args = get_args(hint)
                if Constant in args[1:]:
                    wrapped_type = args[0]
                    assert get_origin(wrapped_type) is Value
                    yield name, get_args(wrapped_type)[0], True
                    continue
            elif get_origin(hint) is Value:
                yield name, get_args(hint)[0], False
            elif get_origin(hint) is not ClassVar:
                yield name, None, False


T = TypeVar("T", bound=Type)


class Memory(Generic[T]):
    """Typed memory buffer — the ABI for a type."""

    type: T
    buffer: bytearray

    def __init__(self, type: T, buffer: bytearray | None = None) -> None:
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
