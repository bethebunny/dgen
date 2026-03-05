"""AST types for .dgen dialect specification files."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ImportDecl:
    """An import statement: from module import Name1, Name2."""

    module: str
    names: list[str]


@dataclass
class TypeRef:
    """A reference to a type: Name, Name<args>, list<T>, or Type."""

    name: str
    args: list[TypeRef] = field(default_factory=list)


@dataclass
class ParamDecl:
    """A compile-time parameter: name: Type [= default]."""

    name: str
    type: TypeRef
    default: str | None = None
    variadic: bool = False


@dataclass
class OperandDecl:
    """A runtime operand: name[: Type] [= default]."""

    name: str
    type: TypeRef | None = None
    default: str | None = None
    variadic: bool = False


@dataclass
class DataField:
    """A data field declaration on a type: name: TypeExpr.

    Describes the type's memory layout using type references.
    E.g. ``dims: Array<Index, rank>`` or ``data: FatPointer<Byte>``.
    """

    name: str
    type: TypeRef


@dataclass
class TraitDecl:
    """A trait declaration: trait Name."""

    name: str


@dataclass
class TypeDecl:
    """A type declaration with optional params and data fields."""

    name: str
    params: list[ParamDecl] = field(default_factory=list)
    data: DataField | None = None
    layout: str | None = None


@dataclass
class OpDecl:
    """An op declaration with params, operands, return type, and blocks."""

    name: str
    params: list[ParamDecl] = field(default_factory=list)
    operands: list[OperandDecl] = field(default_factory=list)
    return_type: TypeRef = field(default_factory=lambda: TypeRef("Type"))
    blocks: list[str] = field(default_factory=list)


@dataclass
class DgenFile:
    """A complete .dgen file."""

    imports: list[ImportDecl] = field(default_factory=list)
    traits: list[TraitDecl] = field(default_factory=list)
    types: list[TypeDecl] = field(default_factory=list)
    ops: list[OpDecl] = field(default_factory=list)
