"""Shared utilities for C literal parsing."""

from __future__ import annotations


_CHAR_ESCAPES: dict[str, int] = {
    "n": 10,
    "t": 9,
    "r": 13,
    "0": 0,
    "\\": 92,
    "'": 39,
    '"': 34,
    "a": 7,
    "b": 8,
    "f": 12,
}

_CONST_BINOPS: dict[str, object] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a // b if b else 0,
    "%": lambda a, b: a % b if b else 0,
    "<<": lambda a, b: a << b,
    ">>": lambda a, b: a >> b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
}

_CONST_UNARY: dict[str, object] = {
    "-": lambda v: -v,
    "+": lambda v: v,
    "~": lambda v: ~v,
    "!": lambda v: int(not v),
}


def parse_c_int(text: str) -> int:
    """Parse a C integer literal string (handles hex, octal, decimal)."""
    s = text.rstrip("uUlL")
    if len(s) > 1 and s[0] == "0" and s[1:].isdigit():
        return int(s, 8)
    return int(s, 0)


def parse_c_char(text: str) -> int:
    """Parse a C character literal ('a', '\\n', etc.) to its integer value."""
    ch = text[1:-1]
    if ch.startswith("\\"):
        esc = ch[1]
        if esc in _CHAR_ESCAPES:
            return _CHAR_ESCAPES[esc]
        if esc == "x":
            return int(ch[2:], 16)
        if esc.isdigit():
            return int(ch[1:], 8)
        raise ValueError(f"unknown escape sequence: {ch}")
    return ord(ch)
