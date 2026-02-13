"""Toy source language AST types.

Uses an arena pattern: expressions are stored in a flat list and referenced
by index, avoiding OwnedPointer and keeping all types Copyable for List.
"""

from utils import Variant
from collections import Optional


# ===----------------------------------------------------------------------=== #
# Expressions
# ===----------------------------------------------------------------------=== #

comptime AnyExpr = Variant[NumberLiteral, TensorLiteral, VarRef, BinaryOp, CallExpr, PrintExpr]


@fieldwise_init
struct NumberLiteral(Copyable, Movable):
    var value: Float64


@fieldwise_init
struct TensorLiteral(Copyable, Movable):
    """A tensor literal like [[1, 2], [3, 4]].

    Values are flattened, shape records the nesting dimensions.
    """

    var values: List[Float64]
    var shape: List[Int]


@fieldwise_init
struct VarRef(Copyable, Movable):
    var name: String


@fieldwise_init
struct BinaryOp(Copyable, Movable):
    var op: String
    var lhs: Int  # index into ExprArena.exprs
    var rhs: Int


@fieldwise_init
struct CallExpr(Copyable, Movable):
    var callee: String
    var args: List[Int]  # indices into ExprArena.exprs


@fieldwise_init
struct PrintExpr(Copyable, Movable):
    var arg: Int  # index into ExprArena.exprs


# ===----------------------------------------------------------------------=== #
# Expression Arena
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct ExprArena(Copyable, Movable):
    """Flat storage for all expressions, referenced by index."""

    var exprs: List[AnyExpr]

    fn __init__(out self):
        self.exprs = List[AnyExpr]()

    fn add(mut self, var expr: AnyExpr) -> Int:
        var idx = len(self.exprs)
        self.exprs.append(expr^)
        return idx

    fn get(self, idx: Int) -> AnyExpr:
        return self.exprs[idx]


# ===----------------------------------------------------------------------=== #
# Statements
# ===----------------------------------------------------------------------=== #

comptime AnyStmt = Variant[VarDecl, ReturnStmt, ExprStmt]


@fieldwise_init
struct VarDecl(Copyable, Movable):
    """var x [<shape>] = expr"""

    var name: String
    var shape: Optional[List[Int]]
    var value: Int  # index into ExprArena


@fieldwise_init
struct ReturnStmt(Copyable, Movable):
    var value: Optional[Int]  # index into ExprArena, or None


@fieldwise_init
struct ExprStmt(Copyable, Movable):
    var expr: Int  # index into ExprArena


# ===----------------------------------------------------------------------=== #
# Top-level
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct Prototype(Copyable, Movable):
    var name: String
    var params: List[String]


@fieldwise_init
struct Function(Copyable, Movable):
    var proto: Prototype
    var body: List[AnyStmt]


@fieldwise_init
struct ToyModule(Copyable, Movable):
    var functions: List[Function]
    var arena: ExprArena
