"""MLIR ODS (.td) generator, from a dialect's resolved model.

Each construct maps as the design's mapping table says. What has no ODS form is
written as a `// dgen:` comment at the definition it belongs to.
"""

from __future__ import annotations

from collections.abc import Iterator

from dgen.gen import model

type Decl = model.Type | model.Trait


def _camel(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_"))


def _def(decl: model.Type) -> str:
    return f"{_camel(decl.dialect)}_{decl.name}"


def _type_interface(trait: model.Trait) -> str:
    return f"{_camel(trait.dialect)}_{trait.name}TypeInterface"


def _op_interface(trait: model.Trait) -> str:
    return f"{_camel(trait.dialect)}_{trait.name}OpInterface"


def _source(arg: model.TypeExpr | model.ParamRef | model.Literal | model.Layout) -> str:
    """The dgen spelling of a resolved reference, qualified, for comments."""
    match arg:
        case model.TypeExpr(decl=None):
            return "Type"
        case model.TypeExpr(decl=model.Type() | model.Trait() as decl, args=args):
            name = f"{decl.dialect}.{decl.name}"
            return f"{name}<{', '.join(map(_source, args))}>" if args else name
        case model.Layout(name=name, args=args):
            return f"{name}<{', '.join(map(_source, args))}>" if args else name
        case model.ParamRef(name=name):
            return name
        case model.Literal(value=value):
            return repr(value)
    raise TypeError(f"no dgen spelling for {arg!r}")


def _constraint(expr: model.TypeExpr | None) -> str:
    decl = None if expr is None else expr.decl
    if decl is None:
        return "AnyType"
    return _type_interface(decl) if isinstance(decl, model.Trait) else _def(decl)


def _narrowed(what: str, expr: model.TypeExpr | None) -> list[str]:
    """A note when ODS checks a type's name but not its arguments."""
    if expr is None or not expr.args:
        return []
    return [f"{what} is {_source(expr)}, checked as {_constraint(expr)} only"]


def _static(static: model.Static) -> str:
    default = "" if static.default is None else f" = {static.default.value!r}"
    return f"static {static.name}: {_source(static.type)}{default} has no ODS form"


def _referenced(dialect: model.Dialect) -> set[str]:
    """The other dialects whose definitions the file names."""

    def exprs(expr: model.TypeExpr | None) -> Iterator[Decl]:
        if expr is not None and expr.decl is not None:
            yield expr.decl

    def decls() -> Iterator[Decl]:
        for t in dialect.types:
            yield from t.traits
            for p in t.params:
                yield from exprs(p.type)
                yield from exprs(p.default)
        for op in dialect.ops:
            yield from op.traits
            yield from exprs(op.result)
            for p in op.params:
                yield from exprs(p.type)
            for o in op.operands:
                yield from exprs(o.type)
            for c in op.constraints:
                match c:
                    case model.HasTrait(trait=trait):
                        yield trait
                    case model.HasType(type=expr):
                        yield from exprs(expr)

    return {d.dialect for d in decls()} - {dialect.name}


def _list(keyword: str, kind: str, items: list[str]) -> Iterator[str]:
    if len(items) == 1:
        yield f"  let {keyword} = ({kind} {items[0]});"
        return
    yield f"  let {keyword} = ({kind}"
    yield from (f"    {item}," for item in items[:-1])
    yield f"    {items[-1]}"
    yield "  );"


def _trait(trait: model.Trait) -> Iterator[str]:
    namespace = f"::dgen::{trait.dialect}"
    yield from (f"// dgen: {_static(s)}" for s in trait.statics)
    for kind in ["Type", "Op"]:
        cpp_name = f"{trait.name}{kind}Interface"
        yield f'def {_camel(trait.dialect)}_{cpp_name} : {kind}Interface<"{cpp_name}"> {{'
        yield f'  let cppNamespace = "{namespace}";'
        yield f'  let description = "dgen trait {trait.dialect}.{trait.name}";'
        yield "}"
        yield ""


def _type_parameter(param: model.Param, notes: list[str]) -> str:
    if not param.holds_type:
        if param.default is not None:
            notes.append(f"default of {param.name} has no ODS form")
        summary = f"a compile-time {_source(param.type)} value"
        return f'AttrOrTypeParameter<"::mlir::TypedAttr", "{summary}">:${param.name}'
    match param.default:
        case model.TypeExpr(decl=model.Type(params=()) as t):
            value = f"::dgen::{t.dialect}::{t.name}Type::get($_ctxt)"
            return f'DefaultValuedParameter<"::mlir::Type", "{value}">:${param.name}'
        case model.TypeExpr() as default:
            notes.append(f"default {_source(default)} of {param.name} has no ODS form")
    if isinstance(param.type.decl, model.Trait):
        summary = f"a type with trait {_source(param.type)}"
        return f'AttrOrTypeParameter<"::mlir::Type", "{summary}">:${param.name}'
    return f'"::mlir::Type":${param.name}'


def _assembly_format(params: tuple[model.Param, ...]) -> str:
    parts = [
        f"{'' if i == 0 else '`,` '}${p.name}"
        if p.default is None
        else f"({'' if i == 0 else '`,` '}${p.name}^)?"
        for i, p in enumerate(params)
    ]
    return f"`<` {' '.join(parts)} `>`"


def _type(t: model.Type) -> Iterator[str]:
    notes: list[str] = []
    parameters = [_type_parameter(p, notes) for p in t.params]
    notes += [_static(s) for s in t.statics]
    notes += [
        f"requires {_constraint_text(c)} has no ODS form on a type"
        for c in t.constraints
    ]
    layout = (
        f"layout {t.layout}"
        if t.layout is not None
        else "; ".join(f"field {f.name}: {_source(f.type)}" for f in t.data)
    )
    traits = ", ".join(_type_interface(trait) for trait in t.traits)
    trait_list = f", [{traits}]" if traits else ""
    yield from (f"// dgen: {note}" for note in notes)
    yield f'def {_def(t)} : TypeDef<{_camel(t.dialect)}_Dialect, "{t.name}"{trait_list}> {{'
    yield f'  let mnemonic = "{t.name}";'
    yield f'  let summary = "{t.dialect}.{t.name}";'
    if layout:
        yield f'  let description = "{layout}";'
    if parameters:
        yield from _list("parameters", "ins", parameters)
        yield f'  let assemblyFormat = "{_assembly_format(t.params)}";'
    yield "}"
    yield ""


def _constraint_text(c: model.Constraint) -> str:
    match c:
        case model.HasTrait(subject=subject, trait=trait):
            return f"{subject} has trait {trait.dialect}.{trait.name}"
        case model.HasType(subject=subject, type=expr):
            return f"{subject} has type {_source(expr)}"
        case model.Expression(text=text):
            return text


def _attribute(param: model.Param, notes: list[str]) -> str:
    if param.default is not None:
        notes.append(f"default of parameter {param.name} has no ODS form")
    match param.type.decl:
        case None:
            return f"TypeAttr:${param.name}"
        case model.Trait() as trait:
            return f"TypeAttrOf<{_type_interface(trait)}>:${param.name}"
    notes += _narrowed(f"parameter {param.name}", param.type)
    return f"Dgen_ValueAttr<{_constraint(param.type)}>:${param.name}"


def _operand(operand: model.Operand, notes: list[str]) -> str:
    if operand.default is not None:
        notes.append(f"default of operand {operand.name} has no ODS form")
    notes += _narrowed(f"operand {operand.name}", operand.type)
    return f"{_constraint(operand.type)}:${operand.name}"


def _op_constraint(
    c: model.Constraint, operands: set[str], notes: list[str]
) -> list[str]:
    match c:
        case model.HasTrait(subject=subject, trait=trait) if subject in operands:
            return [f'TypeIs<"{subject}", {_type_interface(trait)}>']
        case model.HasType(subject=subject, type=expr) if subject in operands:
            notes += _narrowed(f"requires {subject} has type", expr)
            return [f'TypeIs<"{subject}", {_constraint(expr)}>']
    notes.append(f"requires {_constraint_text(c)} has no ODS form")
    return []


def _op(op: model.Op) -> Iterator[str]:
    notes: list[str] = []
    arguments = [_attribute(p, notes) for p in op.params]
    arguments += [_operand(o, notes) for o in op.operands]
    operands = {o.name for o in op.operands}
    traits = [_op_interface(t) for t in op.traits]
    traits += [t for c in op.constraints for t in _op_constraint(c, operands, notes)]
    result = (
        "Dgen_Type" if op.result == model.TypeExpr(None) else _constraint(op.result)
    )
    notes += _narrowed("the result", op.result)
    name = f"{_camel(op.dialect)}_{_camel(op.name)}Op"
    trait_list = f", [{', '.join(traits)}]" if traits else ""
    yield from (f"// dgen: {note}" for note in notes)
    yield f'def {name} : {_camel(op.dialect)}_Op<"{op.name}"{trait_list}> {{'
    yield f'  let summary = "{op.dialect}.{op.name}";'
    if arguments:
        yield from _list("arguments", "ins", arguments)
    yield f"  let results = (outs {result}:$result);"
    if op.blocks:
        yield from _list(
            "regions", "region", [f"SizedRegion<1>:${b}" for b in op.blocks]
        )
    yield "}"
    yield ""


def generate_td(dialect: model.Dialect, *, prefix: str = "dgen_") -> str:
    """Generate an ODS file for a resolved dialect."""
    camel = _camel(dialect.name)
    guard = f"DGEN_{dialect.name.upper()}_TD"
    lines = [
        f"// GENERATED by dgen from {dialect.name}.dgen — do not edit.",
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        'include "mlir/IR/OpBase.td"',
        'include "mlir/IR/AttrTypeBase.td"',
        'include "dgen.td"',
        *(f'include "{name}.td"' for name in sorted(_referenced(dialect))),
        "",
        f"def {camel}_Dialect : Dialect {{",
        f'  let name = "{prefix}{dialect.name}";',
        f'  let cppNamespace = "::dgen::{dialect.name}";',
        *(["  let useDefaultTypePrinterParser = 1;"] if dialect.types else []),
        "}",
        "",
        f"class {camel}_Op<string mnemonic, list<Trait> traits = []> :",
        f"    Op<{camel}_Dialect, mnemonic, traits>;",
        "",
    ]
    for trait in dialect.traits:
        lines.extend(_trait(trait))
    for t in dialect.types:
        lines.extend(_type(t))
    for op in dialect.ops:
        lines.extend(_op(op))
    lines += [f"#endif // {guard}"]
    return "\n".join(lines) + "\n"
