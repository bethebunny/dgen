"""Toy source language AST types.

Python version: uses direct recursive dataclasses instead of the arena pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ===----------------------------------------------------------------------=== #
# Expressions
# ===----------------------------------------------------------------------=== #


@dataclass
class NumberLiteral:
    value: float


@dataclass
class TensorLiteral:
    """A tensor literal like [[1, 2], [3, 4]].

    Values are flattened, shape records the nesting dimensions.
    """

    values: list[float]
    shape: list[int]


@dataclass
class VarRef:
    name: str


@dataclass
class BinaryOp:
    op: str
    lhs: Expression
    rhs: Expression


@dataclass
class CallExpr:
    callee: str
    args: list[Expression]


@dataclass
class PrintExpr:
    arg: Expression


@dataclass
class GradExpr:
    """grad(f) — symbolic gradient: returns the gradient function of f."""

    callee: str


@dataclass
class ApplyExpr:
    """expr(args...) — call the result of an expression (e.g. grad(f)(x))."""

    callee: Expression
    args: list[Expression]


Expression = Union[
    NumberLiteral,
    TensorLiteral,
    VarRef,
    BinaryOp,
    CallExpr,
    PrintExpr,
    GradExpr,
    ApplyExpr,
]


# ===----------------------------------------------------------------------=== #
# Statements
# ===----------------------------------------------------------------------=== #


@dataclass
class VarDecl:
    """var x [<shape>] = expr"""

    name: str
    shape: list[int] | None
    value: Expression


@dataclass
class ReturnStmt:
    value: Expression | None


@dataclass
class ExprStmt:
    expr: Expression


Statement = Union[VarDecl, ReturnStmt, ExprStmt]


# ===----------------------------------------------------------------------=== #
# Top-level
# ===----------------------------------------------------------------------=== #


@dataclass
class Prototype:
    name: str
    params: list[str]


@dataclass
class Function:
    proto: Prototype
    body: list[Statement]


@dataclass
class ToyModule:
    functions: list[Function]
