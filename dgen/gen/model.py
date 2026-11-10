"""A dialect resolved from its .dgen AST, in no target language.

Generators read this model instead of the AST or the built Python classes.

Invariants:
  - every type reference is a TypeExpr bound to its declaration, never a name
  - a re-exported name resolves to the declaration in its defining dialect
  - declarations keep file order, which is asm order
"""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from functools import cached_property

from dgen.gen import ast
from dgen.gen.build import _CONSTRUCTOR_LAYOUTS, _PRIMITIVE_LAYOUTS


@dataclass(frozen=True)
class Literal:
    value: int | float | str | bool


@dataclass(frozen=True)
class ParamRef:
    """A use of a parameter of the enclosing declaration, eg. rank in Array<Index, rank>."""

    name: str


@dataclass(frozen=True)
class TypeExpr:
    """A resolved type reference. A decl of None is the metatype Type."""

    decl: Type | Trait | None
    args: tuple[TypeExpr | ParamRef | Literal, ...] = ()


@dataclass(frozen=True)
class Param:
    name: str
    type: TypeExpr
    default: TypeExpr | None = None

    @property
    def holds_type(self) -> bool:
        return self.type.decl is None or isinstance(self.type.decl, Trait)


@dataclass(frozen=True)
class Operand:
    name: str
    type: TypeExpr | None = None
    default: TypeExpr | None = None


@dataclass(frozen=True)
class Static:
    name: str
    type: TypeExpr
    default: Literal | None = None


@dataclass(frozen=True)
class Layout:
    """A layout named in a data field, eg. Array in Array<Index, rank>, or Byte."""

    name: str
    args: tuple[Layout | TypeExpr | ParamRef | Literal, ...] = ()


@dataclass(frozen=True)
class Field:
    """A data field, part of a type's layout."""

    name: str
    type: Layout | TypeExpr | ParamRef


@dataclass(frozen=True)
class HasTrait:
    subject: str
    trait: Trait


@dataclass(frozen=True)
class HasType:
    subject: str
    type: TypeExpr


@dataclass(frozen=True)
class Expression:
    text: str


type Constraint = HasTrait | HasType | Expression


@dataclass(frozen=True)
class Trait:
    dialect: str
    module: str
    name: str
    statics: tuple[Static, ...] = ()


@dataclass(frozen=True)
class Type:
    dialect: str
    module: str
    name: str
    params: tuple[Param, ...] = ()
    layout: str | None = None
    data: tuple[Field, ...] = ()
    traits: tuple[Trait, ...] = ()
    statics: tuple[Static, ...] = ()
    constraints: tuple[Constraint, ...] = ()


@dataclass(frozen=True)
class Op:
    dialect: str
    module: str
    name: str
    params: tuple[Param, ...] = ()
    operands: tuple[Operand, ...] = ()
    result: TypeExpr | None = None
    blocks: tuple[str, ...] = ()
    traits: tuple[Trait, ...] = ()
    constraints: tuple[Constraint, ...] = ()

    @property
    def fixed_type(self) -> TypeExpr | None:
        """The result type the declaration alone determines, or None if a caller passes it."""
        match self.result:
            case TypeExpr(decl=None):
                return self.result
            case TypeExpr(decl=Type(params=params), args=()) if all(
                p.default is not None for p in params
            ):
                return self.result
            case TypeExpr(decl=Type(params=params), args=args) if (
                args
                and len(args) == len(params)
                and all(isinstance(a, Literal) for a in args)
            ):
                return self.result
        return None


@dataclass(frozen=True)
class Dialect:
    """A resolved .dgen file.

    Invariants:
      - names holds what `from M import ...` lines bind, aliases what `import M` lines bind
      - own traits and types shadow imported names, as in the built module
    """

    name: str
    module: str
    names: dict[str, Type | Trait] = field(default_factory=dict)
    aliases: dict[str, Dialect] = field(default_factory=dict)
    traits: tuple[Trait, ...] = ()
    types: tuple[Type, ...] = ()
    ops: tuple[Op, ...] = ()

    @cached_property
    def scope(self) -> dict[str, Type | Trait]:
        return self.names | {d.name: d for d in (*self.traits, *self.types)}

    def lookup(self, name: str) -> Type | Trait:
        alias, dot, attr = name.partition(".")
        if dot:
            return self.aliases[alias].lookup(attr)
        if (decl := self.scope.get(name)) is None:
            raise NameError(f"{self.name}: {name!r} isn't declared or imported")
        return decl

    @classmethod
    def from_ast(
        cls,
        file: ast.DgenFile,
        *,
        name: str,
        module: str,
        imported: Callable[[str], Dialect],
    ) -> Dialect:
        names: dict[str, Type | Trait] = {}
        aliases: dict[str, Dialect] = {}
        for imp in file.imports:
            dep = imported(imp.module)
            if imp.names:
                names |= {n: dep.lookup(n) for n in imp.names}
            else:
                aliases[imp.module] = dep
        # Declarations resolve against those above them, as in build().
        partial = cls(name=name, module=module, names=names, aliases=aliases)
        resolver = _Resolver(partial)
        for td in file.traits:
            resolver.add(resolver.trait(td))
        for td in file.types:
            resolver.add(resolver.type(td))
        return cls(
            name=name,
            module=module,
            names=names,
            aliases=aliases,
            traits=tuple(resolver.traits),
            types=tuple(resolver.types),
            ops=tuple(resolver.op(od) for od in file.ops),
        )


class _Resolver:
    """Resolves one file's declarations in order against a growing scope."""

    def __init__(self, dialect: Dialect) -> None:
        self.dialect = dialect
        self.traits: list[Trait] = []
        self.types: list[Type] = []

    def add(self, decl: Type | Trait) -> None:
        match decl:
            case Trait():
                self.traits.append(decl)
            case Type():
                self.types.append(decl)
        self.dialect = Dialect(
            name=self.dialect.name,
            module=self.dialect.module,
            names=self.dialect.names,
            aliases=self.dialect.aliases,
            traits=tuple(self.traits),
            types=tuple(self.types),
        )

    def arg(
        self, ref: ast.TypeRef, params: Collection[str]
    ) -> TypeExpr | ParamRef | Literal:
        if ref.name in params and not ref.args:
            return ParamRef(ref.name)
        try:
            return self.expr(ref, params)
        except (NameError, KeyError):
            pass
        try:
            return Literal(literal_eval(ref.name))
        except (ValueError, SyntaxError):
            raise NameError(
                f"{self.dialect.name}: {ref.name!r} is no type, parameter or literal"
            ) from None

    def expr(self, ref: ast.TypeRef, params: Collection[str] = ()) -> TypeExpr:
        decl = None if ref.name == "Type" else self.dialect.lookup(ref.name)
        return TypeExpr(decl, tuple(self.arg(a, params) for a in ref.args))

    def layout(
        self, ref: ast.TypeRef, params: Collection[str]
    ) -> Layout | TypeExpr | ParamRef | Literal:
        """A data field's type, where layout names win over types, as in build()."""
        if ref.name in _CONSTRUCTOR_LAYOUTS:
            return Layout(ref.name, tuple(self.layout(a, params) for a in ref.args))
        if ref.name in _PRIMITIVE_LAYOUTS:
            return Layout(ref.name)
        return self.arg(ref, params)

    def field(self, f: ast.DataField, params: Collection[str]) -> Field:
        match self.layout(f.type, params):
            case Literal() as literal:
                raise NameError(
                    f"{self.dialect.name}: data field {f.name} is {literal}"
                )
            case layout:
                return Field(f.name, layout)

    def trait_ref(self, name: str) -> Trait:
        decl = self.dialect.lookup(name)
        if not isinstance(decl, Trait):
            raise TypeError(f"{self.dialect.name}: {name!r} isn't a trait")
        return decl

    def params(self, decls: list[ast.ParamDecl]) -> tuple[Param, ...]:
        return tuple(
            Param(
                p.name,
                self.expr(p.type),
                None if p.default is None else self.expr(ast.TypeRef(p.default)),
            )
            for p in decls
        )

    def statics(self, decls: list[ast.StaticField]) -> tuple[Static, ...]:
        return tuple(
            Static(
                s.name,
                self.expr(s.type),
                None if s.default is None else Literal(literal_eval(s.default)),
            )
            for s in decls
        )

    def constraints(
        self, decls: list[ast.Constraint], params: Collection[str]
    ) -> tuple[Constraint, ...]:
        def resolved(c: ast.Constraint) -> Constraint:
            match c:
                case ast.HasTraitConstraint(lhs=lhs, trait=trait):
                    return HasTrait(lhs, self.trait_ref(trait))
                case ast.HasTypeConstraint(lhs=lhs, type=type_):
                    return HasType(lhs, self.expr(type_, params))
                case ast.ExpressionConstraint(expr=text):
                    return Expression(text)
            raise TypeError(f"unknown constraint {c!r}")

        return tuple(resolved(c) for c in decls)

    def trait(self, td: ast.TraitDecl) -> Trait:
        d = self.dialect
        return Trait(d.name, d.module, td.name, self.statics(td.statics))

    def type(self, td: ast.TypeDecl) -> Type:
        d = self.dialect
        params = {p.name for p in td.params}
        return Type(
            d.name,
            d.module,
            td.name,
            params=self.params(td.params),
            layout=td.layout,
            data=tuple(self.field(f, params) for f in td.data),
            traits=tuple(self.trait_ref(t) for t in td.traits),
            statics=self.statics(td.statics),
            constraints=self.constraints(td.constraints, params),
        )

    def op(self, od: ast.OpDecl) -> Op:
        d = self.dialect
        params = {p.name for p in od.params}
        return Op(
            d.name,
            d.module,
            od.name,
            params=self.params(od.params),
            operands=tuple(
                Operand(
                    o.name,
                    None if o.type is None else self.expr(o.type, params),
                    None if o.default is None else self.expr(ast.TypeRef(o.default)),
                )
                for o in od.operands
            ),
            result=None
            if od.return_type is None
            else self.expr(od.return_type, params),
            blocks=tuple(od.blocks),
            traits=tuple(self.trait_ref(t) for t in od.traits),
            constraints=self.constraints(od.constraints, params),
        )
