from __future__ import annotations

import ctypes
from copy import deepcopy as _deepcopy
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterator, Self, TypeVar

from .layout import BYTE, Array, FatPointer, Layout, Pointer

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


def _bytearray_address(buf: bytearray) -> int:
    """Get the raw ctypes address of a bytearray."""
    ct = (ctypes.c_char * len(buf)).from_buffer(buf)
    return ctypes.addressof(ct)


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

        Works for all layouts: inline scalars/arrays pack directly into the
        buffer; FatPointer allocates a backing origin and packs {ptr, length}.
        """
        if isinstance(value, str):
            value = value.encode("utf-8")

        layout = type.__layout__

        if isinstance(layout, FatPointer):
            return cls._from_fat_pointer(type, layout, value)

        parsed = layout.parse(value)
        mem = cls(type)
        if isinstance(parsed, list):
            mem.pack(*parsed)
        else:
            mem.pack(parsed)
        return mem

    @classmethod
    def _from_fat_pointer(cls, type: Type, layout: FatPointer, value: object) -> Memory:
        """Create Memory for a FatPointer layout with a backing origin."""
        if isinstance(value, bytes):
            origin = bytearray(value)
            length = len(value)
        elif isinstance(value, list):
            pointee = layout.pointee
            length = len(value)
            origin = bytearray(pointee.struct.size * length)
            inner_origins: list[bytearray] = []
            if isinstance(pointee, (FatPointer, Pointer)):
                # Nested pointer: recursively create Memory for each element
                elem_type: Type = type.element_type  # type: ignore[attr-defined]
                for i, v in enumerate(value):
                    elem_mem = cls.from_value(elem_type, v)
                    offset = i * pointee.struct.size
                    origin[offset : offset + pointee.struct.size] = elem_mem.buffer
                    inner_origins.extend(elem_mem.origins)
            else:
                for i, v in enumerate(value):
                    parsed = pointee.parse(v)
                    pointee.struct.pack_into(origin, i * pointee.struct.size, parsed)
        else:
            raise ValueError(
                f"cannot create FatPointer from {value.__class__.__name__}"
            )

        mem = cls(type)
        mem.origins.append(origin)
        if isinstance(value, list):
            mem.origins.extend(inner_origins)
        mem.pack(_bytearray_address(origin), length)
        return mem

    def to_python(self) -> object:
        """Convert back to a Python value by reading from the buffer.

        For pointer layouts, dereferences the packed pointer to read data.
        """
        layout = self.layout

        if isinstance(layout, FatPointer):
            ptr, length = self.unpack()
            pointee = layout.pointee
            ps = pointee.struct.size
            data = bytes((ctypes.c_char * (length * ps)).from_address(ptr))
            if pointee is BYTE:
                return data.decode("utf-8")
            if isinstance(pointee, (FatPointer, Pointer)):
                # Nested pointer: recursively convert each element
                elem_type: Type = self.type.element_type  # type: ignore[attr-defined]
                return [
                    Memory(
                        elem_type, bytearray(data[i * ps : (i + 1) * ps])
                    ).to_python()
                    for i in range(length)
                ]
            return [pointee.struct.unpack_from(data, i * ps)[0] for i in range(length)]

        if isinstance(layout, Array):
            return list(self.unpack())

        vals = self.unpack()
        return vals[0] if len(vals) == 1 else vals

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
