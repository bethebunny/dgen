"""FFI bridge for register-passable layouts.

Maps a layout's struct.Struct format to LLVM IR type strings, ctypes
types, and LLVM constant literals. Only handles register-passable
layouts — callers must normalize non-register-passable types to
by-pointer (c_void_p / "ptr" / address) before calling.
"""

from __future__ import annotations

import ctypes
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dgen.layout import Layout

_LLVM = {"q": "i64", "Q": "i64", "d": "double", "B": "i1", "P": "ptr", "s": "i8"}
_CTYPES = {
    "q": ctypes.c_int64,
    "Q": ctypes.c_uint64,
    "d": ctypes.c_double,
    "B": ctypes.c_bool,
    "P": ctypes.c_void_p,
    "s": ctypes.c_uint8,
}
_AGGREGATE_CACHE: dict[str, type[ctypes.Structure]] = {}
_FORMAT_RE = re.compile(r"(\d*)([a-zA-Z])")


def _struct_fields(fmt: str) -> list[str]:
    """``"2qP"`` → ``["q", "q", "P"]``."""
    return [ch for count, ch in _FORMAT_RE.findall(fmt) for _ in range(int(count or 1))]


def llvm_type(layout: Layout) -> str:
    """LLVM IR type string for a register-passable layout."""
    fields = [_LLVM.get(f, "i8") for f in _struct_fields(layout.struct.format)]
    match fields:
        case []:
            return "void"
        case [single]:
            return single
        case _:
            return "{ " + ", ".join(fields) + " }"


def _format_llvm_value(ty: str, value: object) -> str:
    if ty == "ptr":
        return f"ptr inttoptr (i64 {value} to ptr)" if value else "ptr null"
    if isinstance(value, float):
        n = int(value)
        return f"{ty} {n}.0" if float(n) == value else f"{ty} {value}"
    return f"{ty} {value}"


def llvm_constant(buffer: bytes, layout: Layout) -> str:
    """Format buffer bytes as an LLVM literal for a register-passable layout."""
    fields = _struct_fields(layout.struct.format)
    values = layout.struct.unpack(buffer)
    parts = [_format_llvm_value(_LLVM.get(f, "i8"), v) for f, v in zip(fields, values)]
    match parts:
        case []:
            return "undef"
        case [single]:
            return single.split(" ", 1)[1]
        case _:
            return "{ " + ", ".join(parts) + " }"


def ctype(layout: Layout) -> type[ctypes._CData]:
    """ctypes type for a register-passable layout.

    Aggregate Structure classes are cached because ctypes checks class
    identity when matching argument types.
    """
    fields = _struct_fields(layout.struct.format)
    match fields:
        case []:
            return ctypes.c_void_p
        case [single]:
            return _CTYPES.get(single, ctypes.c_uint8)
        case _:
            fmt = layout.struct.format
            if fmt not in _AGGREGATE_CACHE:
                ct_fields = [
                    (f"_{i}", _CTYPES.get(f, ctypes.c_uint8))
                    for i, f in enumerate(fields)
                ]
                _AGGREGATE_CACHE[fmt] = type(
                    f"_S{len(fields)}", (ctypes.Structure,), {"_fields_": ct_fields}
                )
            return _AGGREGATE_CACHE[fmt]


def to_ffi(layout: Layout, buffer: bytearray) -> object:
    """Convert Memory bytes to an FFI-passable value for a register-passable layout.

    Scalars → plain Python values (ctypes callbacks require this).
    Aggregates → ctypes.Structure instances.
    """
    val = ctype(layout).from_buffer_copy(buffer)
    return val if isinstance(val, ctypes.Structure) else val.value


def from_ffi(layout: Layout, raw: object) -> bytearray:
    """Convert an FFI return value back to Memory bytes for a register-passable layout.

    Structures → copy bytes. Scalars → pack via ctypes.
    """
    if isinstance(raw, ctypes.Structure):
        return bytearray(bytes(raw))
    raw_bytes = bytes(ctype(layout)(raw))
    return bytearray(raw_bytes[: layout.byte_size])
