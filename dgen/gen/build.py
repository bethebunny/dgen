"""Runtime module builder for .dgen dialect specifications.

Creates Python classes directly via metaprogramming instead of generating and
exec()-ing source code.  Used by the DgenLoader import hook.
"""

from __future__ import annotations

import ast as _ast
import builtins
import dataclasses
import importlib
from collections.abc import Callable
from types import ModuleType

import dgen
from dgen import Block, Dialect, Op, Type, TypeType, Value, layout
from dgen.gen.ast import (
    Constraint,
    DgenFile,
    OpDecl,
    TraitDecl,
    TypeDecl,
    TypeRef,
)

# Layout functions return either a Layout (for Type params) or a JSON scalar
# (for value params like Index used as Array size arguments).
_LayoutFn = Callable[[object], object]

_PRIMITIVE_LAYOUTS: dict[str, layout.Layout] = {
    "Int": layout.Int(),
    "Float64": layout.Float64(),
    "Void": layout.Void(),
    "Byte": layout.Byte(),
    "String": layout.String(),
}

_CONSTRUCTOR_LAYOUTS: dict[str, type[layout.Layout]] = {
    "Pointer": layout.Pointer,
    "Array": layout.Array,
    "Span": layout.Span,
    "Record": layout.Record,
}


def _op_class_name(asm_name: str) -> str:
    return "".join(p.capitalize() for p in asm_name.split("_")) + "Op"


def _resolve_type(name: str, ns: dict[str, object]) -> type:
    """Look up a class from the namespace, supporting qualified names like ``mod.Cls``."""
    mod, sep, attr = name.partition(".")
    return getattr(ns[mod], attr) if sep else ns[name]  # type: ignore[return-value]


def _resolve_param_type(ref: TypeRef, ns: dict[str, object]) -> type[Type]:
    """Convert a TypeRef to its actual Python Type class (AST boundary)."""
    if ref.name == "Type":
        return TypeType
    return _resolve_type(ref.name, ns)  # type: ignore[return-value]


def _is_type_kinded(param_type: type[Type]) -> bool:
    """True when *param_type* values are themselves types (metatype params)."""
    return param_type is Type or issubclass(param_type, TypeType)


def _is_list_of_types(param_type: type[Type]) -> bool:
    """True when *param_type* holds a list of types (e.g. the builtin List<Type>).

    A type qualifies when it has a type-kinded param AND its layout is
    Span-based (i.e. it stores elements as a variable-length array).
    This distinguishes List<Type> (Span layout) from Function<Type>
    (Void layout).
    """
    params: tuple[tuple[str, type[Type]], ...] = getattr(param_type, "__params__", ())
    if not any(_is_type_kinded(pt) for _, pt in params):
        return False
    try:
        kwargs = {name: TypeType() for name, pt in params if _is_type_kinded(pt)}
        return isinstance(param_type(**kwargs).__layout__, layout.Span)
    except Exception:
        return False


def _ref_has_params(ref: TypeRef, param_names: set[str]) -> bool:
    return ref.name in param_names or any(
        _ref_has_params(a, param_names) for a in ref.args
    )


def _layout_fn(
    ref: TypeRef,
    resolved_params: dict[str, type[Type]],
    ns: dict[str, object],
) -> _LayoutFn:
    """Return a ``(self) -> layout_arg`` closure for any TypeRef.

    For parametric refs the closure reads instance attributes; for fully static
    refs it captures the layout constant at class-build time and ignores *self*.
    """
    if param_type := resolved_params.get(ref.name):
        pname = ref.name
        if _is_type_kinded(param_type):
            return lambda self, _n=pname: (
                dgen.type.type_constant(getattr(self, _n)).__layout__
            )
        return lambda self, _n=pname: getattr(self, _n).__constant__.to_json()
    if ctor := _CONSTRUCTOR_LAYOUTS.get(ref.name):
        sub_fns = [_layout_fn(a, resolved_params, ns) for a in ref.args]
        return lambda self, _c=ctor, _fs=sub_fns: _c(*[f(self) for f in _fs])
    static: object = (
        _PRIMITIVE_LAYOUTS.get(ref.name) or _resolve_type(ref.name, ns).__layout__
    )
    return lambda self, _l=static: _l


def _as_layout_or_property(fn: _LayoutFn, parametric: bool) -> layout.Layout | property:
    """Evaluate *fn* statically (``fn(None)``) or wrap as a property."""
    return property(fn) if parametric else fn(None)  # type: ignore[return-value]


def _fmt_constraint(c: Constraint) -> str:
    if c.kind == "match":
        return f"{c.lhs} ~= {c.pattern}"
    if c.kind == "eq":
        return f"{c.lhs} == {c.rhs}"
    assert c.expr is not None
    return c.expr


def _make_layout(
    td: TypeDecl, ns: dict[str, object]
) -> layout.Layout | property | None:
    """Compute the ``__layout__`` for a type: static Layout, property, or None."""
    resolved_params = {p.name: _resolve_param_type(p.type, ns) for p in td.params}
    param_names = set(resolved_params)

    if td.layout is not None:
        if primitive := _PRIMITIVE_LAYOUTS.get(td.layout):
            return primitive
        ctor = _CONSTRUCTOR_LAYOUTS[td.layout]
        if not td.params:
            return ctor(layout.Void())  # type: ignore[return-value]
        # Record whose layout IS the record built from a list-of-types param
        # (e.g. Tuple<types: List<Type>>: layout Record).
        if ctor is layout.Record and (
            lp := next(
                (p for p in td.params if _is_list_of_types(resolved_params[p.name])),
                None,
            )
        ):
            pname = lp.name
            return property(
                lambda self, _n=pname: layout.Record(
                    [
                        (str(i), dgen.type.type_constant(t).__layout__)
                        for i, t in enumerate(getattr(self, _n))
                    ]
                )
            )
        fns = [_layout_fn(TypeRef(p.name), resolved_params, ns) for p in td.params]
        return property(lambda self, _c=ctor, _fs=fns: _c(*[f(self) for f in _fs]))

    if td.data:
        is_parametric = any(_ref_has_params(df.type, param_names) for df in td.data)
        if len(td.data) == 1:
            return _as_layout_or_property(
                _layout_fn(td.data[0].type, resolved_params, ns), is_parametric
            )
        pairs = [(df.name, _layout_fn(df.type, resolved_params, ns)) for df in td.data]
        return _as_layout_or_property(
            lambda self, _ps=pairs: layout.Record([(n, f(self)) for n, f in _ps]),
            is_parametric,
        )

    return None


def _annotation_for_param(param_type: type[Type]) -> str:
    if _is_list_of_types(param_type):
        return "list[Value[dgen.TypeType]]"
    if _is_type_kinded(param_type):
        return "Value[dgen.TypeType]"
    return f"Value[{param_type.__name__}]"


def _make_type_default(
    ref: TypeRef,
    type_map: dict[str, TypeDecl],
    known_names: set[str],
    ns: dict[str, object],
) -> object | None:
    """Build the default type value for an op's ``type`` field, or ``None``."""
    try:
        if not ref.args:
            td = type_map.get(ref.name)
            can_default = (
                "." not in ref.name
                and ref.name in known_names
                and (td is None or all(p.default is not None for p in td.params))
            )
            return _resolve_type(ref.name, ns)() if can_default else None
        td = type_map.get(ref.name)
        if td is None or len(ref.args) != len(td.params):
            return None
        kwargs = {
            param.name: _resolve_type(param.type.name, ns)().constant(
                _ast.literal_eval(arg.name)
            )
            for arg, param in zip(ref.args, td.params)
        }
        return _resolve_type(ref.name, ns)(**kwargs)
    except (ValueError, KeyError):
        return None


def _build_trait(td: TraitDecl, ns: dict[str, object]) -> type:
    trait_ns: dict[str, object] = {"__module__": ns.get("__name__", "")}
    annotations = {sf.name: sf.type.name for sf in td.statics if sf.default is None}
    trait_ns.update(
        {
            sf.name: _ast.literal_eval(sf.default)
            for sf in td.statics
            if sf.default is not None
        }
    )
    if annotations:
        trait_ns["__annotations__"] = annotations
    return type(td.name, (), trait_ns)


def _build_type(td: TypeDecl, dialect: Dialect, ns: dict[str, object]) -> type:
    resolved_params = {p.name: _resolve_param_type(p.type, ns) for p in td.params}

    type_ns: dict[str, object] = {"__module__": ns.get("__name__", "")}
    annotations: dict[str, str] = {}

    layout_val = _make_layout(td, ns)
    if layout_val is not None:
        type_ns["__layout__"] = layout_val

    annotations.update(
        {p.name: _annotation_for_param(resolved_params[p.name]) for p in td.params}
    )
    type_ns.update(
        {
            p.name: _resolve_type(p.default, ns)()
            for p in td.params
            if p.default is not None
        }
    )
    if td.params:
        type_ns["__params__"] = tuple(
            (p.name, resolved_params[p.name]) for p in td.params
        )

    annotations.update(
        {sf.name: sf.type.name for sf in td.statics if sf.default is None}
    )
    type_ns.update(
        {
            sf.name: _ast.literal_eval(sf.default)
            for sf in td.statics
            if sf.default is not None
        }
    )
    if annotations:
        type_ns["__annotations__"] = annotations

    bases: tuple[type, ...] = tuple(_resolve_type(t, ns) for t in td.traits) + (Type,)
    cls = dataclasses.dataclass(frozen=True, eq=False)(type(td.name, bases, type_ns))
    dialect.type(td.name)(cls)
    return cls


def _build_op(
    od: OpDecl,
    dialect: Dialect,
    ns: dict[str, object],
    type_map: dict[str, TypeDecl],
    known_names: set[str],
) -> type:
    resolved_params = {p.name: _resolve_param_type(p.type, ns) for p in od.params}

    op_ns: dict[str, object] = {"__module__": ns.get("__name__", "")}
    annotations: dict[str, str] = {
        p.name: _annotation_for_param(resolved_params[p.name]) for p in od.params
    }

    annotations.update(
        {
            op.name: f"Value | {op.default}" if op.default else "Value"
            for op in od.operands
        }
    )
    op_ns.update(
        {op.name: _resolve_type(op.default, ns)() for op in od.operands if op.default}
    )

    annotations["type"] = "Type"
    ret = od.return_type
    if ret is not None and ret.name != "Type":
        default = _make_type_default(ret, type_map, known_names, ns)
        if default is not None:
            op_ns["type"] = default

    annotations.update({block_name: "Block" for block_name in od.blocks})

    if od.params:
        op_ns["__params__"] = tuple(
            (p.name, resolved_params[p.name]) for p in od.params
        )
    if od.operands:
        op_ns["__operands__"] = tuple(
            (op.name, _resolve_type(op.type.name, ns) if op.type is not None else Type)
            for op in od.operands
        )
    if od.blocks:
        op_ns["__blocks__"] = tuple(od.blocks)
    if od.constraints:
        op_ns["__constraints__"] = tuple(_fmt_constraint(c) for c in od.constraints)

    op_ns["__annotations__"] = annotations
    bases: tuple[type, ...] = tuple(_resolve_type(t, ns) for t in od.traits) + (Op,)
    cls = dataclasses.dataclass(eq=False, kw_only=True)(
        type(_op_class_name(od.name), bases, op_ns)
    )
    dialect.op(od.name)(cls)
    return cls


def build(
    ast: DgenFile,
    dialect_name: str,
    import_map: dict[str, str],
    module: ModuleType,
) -> None:
    """Build all types, traits, and ops into ``module.__dict__``."""
    ns = module.__dict__

    ns.setdefault("dgen", dgen)
    ns.setdefault("layout", layout)
    ns.setdefault("Type", Type)
    ns.setdefault("Value", Value)
    ns.setdefault("Op", Op)
    ns.setdefault("Block", Block)

    d = Dialect(dialect_name)
    ns[dialect_name] = d

    for imp in ast.imports:
        if py_mod := import_map.get(imp.module):
            mod = importlib.import_module(py_mod)
            if imp.names:
                for name in imp.names:
                    obj = getattr(mod, name)
                    ns[name] = obj
                    # Re-export imported types: register in current dialect so
                    # unqualified lookup (e.g. parser) and formatting work.
                    if isinstance(obj, builtins.type) and issubclass(obj, Type):
                        d.types[name] = obj
            else:
                ns[imp.module] = mod

    for td in ast.traits:
        ns[td.name] = _build_trait(td, ns)

    type_map = {td.name: td for td in ast.types}
    known_names = set(type_map) | {name for imp in ast.imports for name in imp.names}
    for td in ast.types:
        ns[td.name] = _build_type(td, d, ns)

    for od in ast.ops:
        ns[_op_class_name(od.name)] = _build_op(od, d, ns, type_map, known_names)
