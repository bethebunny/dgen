"""Python .pyi stub generator, from a dialect's resolved model.

The import hook loads the ``.dgen`` file and keeps its AST, ``model_of`` resolves
that into a ``model.Dialect``, and this module writes the stub from the model.
Field order and defaults follow what ``build.py`` makes.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

from dgen.gen import model
from dgen.gen.importer import model_of


def _spelled(decl: model.Type | model.Trait, dialect: model.Dialect) -> str:
    """How the stub names a declaration: bare if in scope, else through an alias."""
    if dialect.scope.get(decl.name) == decl:
        return decl.name
    return next(
        (
            f"{alias}.{decl.name}"
            for alias, dep in dialect.aliases.items()
            if dep.scope.get(decl.name) == decl
        ),
        decl.name,
    )


def _value(expr: model.TypeExpr) -> str:
    """A call that builds the type expr names, eg. Int(bits=Index().constant(64))."""
    match expr.decl:
        case None:
            return "dgen.TypeType()"
        case model.Trait(name=name):
            raise TypeError(f"trait {name} has no instances")
        case model.Type(name=name, params=params):
            args = expr.args or tuple(p.default for p in params)
            fields = [f"{p.name}={_argument(p, a)}" for p, a in zip(params, args)]
            return f"{name}({', '.join(fields)})"


def _argument(
    param: model.Param, arg: model.TypeExpr | model.ParamRef | model.Literal | None
) -> str:
    match arg:
        case model.Literal(value=value):
            return f"{_value(param.type)}.constant({value!r})"
        case model.TypeExpr():
            return _value(arg)
    raise ValueError(f"{param.name} has no value")


def _annotation(param: model.Param, dialect: model.Dialect) -> str:
    if param.holds_type:
        return "Value[dgen.TypeType]"
    assert param.type.decl is not None
    return f"Value[{_spelled(param.type.decl, dialect)}]"


def _static_annotation(static: model.Static, dialect: model.Dialect) -> str:
    decl = static.type.decl
    return "Type" if decl is None else _spelled(decl, dialect)


def _operand(operand: model.Operand, dialect: model.Dialect) -> str:
    match operand.default:
        case model.TypeExpr(decl=model.Type() as decl) as default:
            return (
                f"{operand.name}: Value | {_spelled(decl, dialect)} = {_value(default)}"
            )
    return f"{operand.name}: Value"


def _class(header: str, fields: list[str]) -> Iterator[str]:
    yield header
    yield from [f"    {f}" for f in fields] if fields else ["    ..."]
    yield ""


def _bases(traits: tuple[model.Trait, ...], base: str, dialect: model.Dialect) -> str:
    return ", ".join([*(_spelled(t, dialect) for t in traits), base])


def _trait_stub(trait: model.Trait, dialect: model.Dialect) -> Iterator[str]:
    yield from _class(
        f"class {trait.name}(Trait):",
        [
            f"{s.name}: {_static_annotation(s, dialect)}"
            if s.default is None
            else f"{s.name} = {s.default.value!r}"
            for s in trait.statics
        ],
    )


def _type_stub(t: model.Type, dialect: model.Dialect) -> Iterator[str]:
    params = [
        f"{p.name}: {_annotation(p, dialect)}"
        + ("" if p.default is None else f" = {_value(p.default)}")
        for p in t.params
    ]
    # A static with a default is a class attribute, not a dataclass field.
    statics = [
        f"{s.name}: {_static_annotation(s, dialect)}"
        for s in t.statics
        if s.default is None
    ]
    yield "@dataclass(frozen=True, eq=False)"
    yield from _class(
        f"class {t.name}({_bases(t.traits, 'Type', dialect)}):", params + statics
    )


def _op_class_name(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_")) + "Op"


def _op_stub(op: model.Op, dialect: model.Dialect) -> Iterator[str]:
    # The builder drops op parameter defaults, so the stub does too.
    params = [f"{p.name}: {_annotation(p, dialect)}" for p in op.params]
    operands = [_operand(o, dialect) for o in op.operands]
    fixed = op.fixed_type
    result = ["type: Type" + ("" if fixed is None else f" = {_value(fixed)}")]
    blocks = [f"{b}: Block" for b in op.blocks]
    yield "@dataclass(eq=False, kw_only=True)"
    yield from _class(
        f"class {_op_class_name(op.name)}({_bases(op.traits, 'Op', dialect)}):",
        params + operands + result + blocks,
    )


def generate_pyi(module: ModuleType, dialect_name: str) -> str:
    """Generate a ``.pyi`` type stub for a dialect module loaded from a .dgen file."""
    dialect = model_of(module)

    imported_names: dict[str, list[str]] = {}
    for name, decl in dialect.names.items():
        imported_names.setdefault(decl.module, []).append(name)

    dgen_imports = sorted(
        {"Dialect", "Op", "Type", "Value"}
        | ({"Block"} if any(op.blocks for op in dialect.ops) else set())
        | ({"Trait"} if dialect.traits else set())
    )

    lines: list[str] = [
        f"# GENERATED by dgen from {dialect_name}.dgen — do not edit.",
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
        "import dgen",
        f"from dgen import {', '.join(dgen_imports)}",
        *(
            f"import {dep.module} as {alias}"
            for alias, dep in sorted(dialect.aliases.items())
        ),
        *(
            f"from {src} import {', '.join(sorted(names))}"
            for src, names in sorted(imported_names.items())
        ),
        "",
        f'{dialect_name} = Dialect("{dialect_name}")',
        "",
    ]
    for trait in dialect.traits:
        lines.extend(_trait_stub(trait, dialect))
    for t in dialect.types:
        lines.extend(_type_stub(t, dialect))
    for op in dialect.ops:
        lines.extend(_op_stub(op, dialect))

    return "\n".join(lines) + "\n"
