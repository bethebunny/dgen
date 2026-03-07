from __future__ import annotations

import ctypes
from copy import deepcopy as _deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterator, Self, TypeVar

from .layout import Layout, Record, _bytearray_address
from .layout import String as StringLayout

if TYPE_CHECKING:
    from .value import Constant


class Type:
    """Any dialect type.

    Types registered via @dialect.type() get _asm_name and dialect set
    automatically. The format_expr() function handles formatting via
    type_asm() for registered types, or falls back to .asm for types
    with hand-written formatting (e.g. llvm types).
    """

    __layout__: Layout
    __params__: ClassVar[Fields] = ()

    def constant(self, value: object) -> Constant[Self]:
        """Create a Constant wrapping this type and a Python value."""
        from .value import Constant

        return Constant(type=self, value=Memory.from_value(self, value))

    def as_value(self) -> Constant[Self]:
        """Wrap this type as a Constant[TypeType] value."""
        from .value import Constant

        tt = TypeType(concrete=self)
        return Constant(type=tt, value=Memory.from_json(tt, self._type_to_json()))

    def _type_to_json(self) -> dict[str, object]:
        """Serialize this type to a JSON-compatible dict for TypeType Memory."""
        cls = type(self)
        dialect = getattr(cls, "dialect", None)
        prefix = (
            f"{dialect.name}."
            if dialect is not None and dialect.name != "builtin"
            else ""
        )
        tag = f"{prefix}{getattr(cls, '_asm_name', type(self).__name__)}"
        result: dict[str, object] = {"tag": tag}
        for name, _ in self.__params__:
            val = getattr(self, name)
            if isinstance(val, Type):
                result[name] = val._type_to_json()
            else:
                result[name] = val.__constant__.to_json()
        return result

    @classmethod
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # TypeType's concrete param must stay a bare Type (no infinite wrapping)
        if cls.__name__ == "TypeType":
            return
        type_kinded = [
            name for name, ft in getattr(cls, "__params__", ()) if ft is Type
        ]
        if not type_kinded:
            return

        def _post_init(self: Type, _names: list[str] = type_kinded) -> None:
            for name in _names:
                val = getattr(self, name)
                # Wrap bare Types as Constant[TypeType]. Skip if already wrapped.
                # Bare Types don't have 'value' attr; Constants do.
                if isinstance(val, Type) and not hasattr(val, "value"):
                    object.__setattr__(self, name, val.as_value())

        cls.__post_init__ = _post_init  # type: ignore[attr-defined]

    @property
    def type_layout(self) -> Record:
        """Layout for this type as a value (tag + params)."""
        fields: list[tuple[str, Layout]] = [("tag", StringLayout())]
        for name, _ in self.__params__:
            val = getattr(self, name)
            if isinstance(val, Type):
                # Type-kinded param: use its type_layout recursively
                fields.append((name, val.type_layout))
            else:
                # Value-kinded param: use the param value's type's __layout__
                fields.append((name, val.__constant__.type.__layout__))
        return Record(fields)

    @property
    def parameters(self) -> Iterator[tuple[str, Type]]:
        for name, field in self.__params__:
            yield name, getattr(self, name)


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
