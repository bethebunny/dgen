"""The expression language of `requires` clauses.

expr    = or
or      = and ("or" and)*
and     = not ("and" not)*
not     = "not" not | compare
compare = sum [("==" | "!=" | "<" | "<=" | ">" | ">=") sum]
sum     = term (("+" | "-") term)*
term    = unary (("*" | "//" | "%") unary)*
unary   = "-" unary | path
path    = atom ("." NAME)*
atom    = NAME | "$" NAME | INT | "(" expr ")"
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Name:
    """An operand or parameter."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeOf:
    """The type of an operand or parameter: `$name`."""

    name: str

    def __str__(self) -> str:
        return f"${self.name}"


@dataclass(frozen=True)
class Attribute:
    """A parameter of a type, or of a value's type."""

    value: Expression
    name: str

    def __str__(self) -> str:
        return f"{self.value}.{self.name}"


@dataclass(frozen=True)
class Literal:
    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Unary:
    op: str
    operand: Expression

    def __str__(self) -> str:
        return f"{self.op} {self.operand}" if self.op == "not" else f"-{self.operand}"


@dataclass(frozen=True)
class Binary:
    op: str
    lhs: Expression
    rhs: Expression

    def __str__(self) -> str:
        return f"({self.lhs} {self.op} {self.rhs})"


type Expression = Name | TypeOf | Attribute | Literal | Unary | Binary

_TOKEN = re.compile(r"\s*(?:(\d+)|([A-Za-z_]\w*)|(//|==|!=|<=|>=|[-+*%<>().$]))")
_LEVELS: list[tuple[str, ...]] = [
    ("or",),
    ("and",),
    ("==", "!=", "<", "<=", ">", ">="),
    ("+", "-"),
    ("*", "//", "%"),
]
_COMPARISONS = _LEVELS[2]


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    at = 0
    while text[at:].strip():
        if (match := _TOKEN.match(text, at)) is None:
            raise SyntaxError(f"unexpected {text[at:].strip()[0]!r} in {text!r}")
        tokens.append(match.group(match.lastindex or 0))
        at = match.end()
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokens(text)
        self.at = 0

    def peek(self) -> str | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self) -> str:
        if (token := self.peek()) is None:
            raise SyntaxError(f"unexpected end of {self.text!r}")
        self.at += 1
        return token

    def expect(self, token: str) -> None:
        if (found := self.take()) != token:
            raise SyntaxError(f"expected {token!r}, found {found!r} in {self.text!r}")

    def binary(self, level: int) -> Expression:
        if level == len(_LEVELS):
            return self.unary()
        if _LEVELS[level] == ("and",):
            # `not` binds looser than comparisons and tighter than `and`.
            lhs = self.negation()
            while self.peek() == "and":
                lhs = Binary(self.take(), lhs, self.negation())
            return lhs
        lhs = self.binary(level + 1)
        while self.peek() in _LEVELS[level]:
            lhs = Binary(self.take(), lhs, self.binary(level + 1))
            if _LEVELS[level] is _COMPARISONS and self.peek() in _COMPARISONS:
                raise SyntaxError(f"chained comparison in {self.text!r}")
        return lhs

    def negation(self) -> Expression:
        if self.peek() == "not":
            return Unary(self.take(), self.negation())
        return self.binary(2)

    def unary(self) -> Expression:
        if self.peek() == "-":
            return Unary(self.take(), self.unary())
        value = self.atom()
        while self.peek() == ".":
            self.take()
            value = Attribute(value, self.name())
        return value

    def name(self) -> str:
        token = self.take()
        if not (token[0].isalpha() or token[0] == "_") or token in ("and", "or", "not"):
            raise SyntaxError(f"expected a name, found {token!r} in {self.text!r}")
        return token

    def atom(self) -> Expression:
        match self.peek():
            case "(":
                self.take()
                inner = self.binary(0)
                self.expect(")")
                return inner
            case "$":
                self.take()
                return TypeOf(self.name())
            case str(token) if token.isdigit():
                return Literal(int(self.take()))
            case _:
                return Name(self.name())


def parse_expression(text: str) -> Expression:
    parser = _Parser(text)
    expression = parser.binary(0)
    if (extra := parser.peek()) is not None:
        raise SyntaxError(f"unexpected {extra!r} in {text!r}")
    return expression
