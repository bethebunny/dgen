"""Runtime module builder for .dgen dialect specifications.

Creates Python classes directly via metaprogramming instead of generating and
exec()-ing source code.  Used by the DgenLoader import hook.
"""

from __future__ import annotations

import ast as _ast
import dataclasses
import importlib
from collections.abc import Callable
from types import ModuleType

import dgen
from dgen import Block, Dialect, Op, Type, Value, layout
from dgen.gen.ast import DgenFile, OpDecl, ParamDecl, TraitDecl, TypeDecl, TypeRef

_PRIMITIVE_LAYOUTS: dict[str, layout.Layout] = {
    "Int": layout.Int(),
    "Float64": layout.Float64(),
    "Void": layout.Void(),
    "Byte": layout.Byte(),
    "String": layout.String(),
}

_CONSTRUCTOR_LAYOUTS: dict[str, type] = {
    "Pointer": layout.Pointer,
    "Array": layout.Array,
    "FatPointer": layout.FatPointer,
    "Record": layout.Record,
}


def _op_class_name(asm_name: str) -> str:
    return "".join(p.capitalize() for p in asm_name.split("_")) + "Op"


def _resolve_type(name: str, ns: dict[str, object]) -> type:
    """Look up a class from the module namespace, supporting qualified names."""
    if "." in name:
        mod_name, attr = name.split(".", 1)
        return getattr(ns[mod_name], attr)
    return ns[name]  # type: ignore[return-value]


def _ref_has_params(ref: TypeRef, param_map: dict[str, ParamDecl]) -> bool:
    if ref.name in param_map:
        return True
    return any(_ref_has_params(a, param_map) for a in ref.args)


def _compute_static_layout(ref: TypeRef, ns: dict[str, object]) -> layout.Layout:
    """Compute a Layout at class-build time (fully static, no parameters)."""
    primitive = _PRIMITIVE_LAYOUTS.get(ref.name)
    if primitive is not None:
        return primitive
    layout_cls = _CONSTRUCTOR_LAYOUTS.get(ref.name)
    if layout_cls is not None:
        args = [_compute_static_layout(a, ns) for a in ref.args]
        return layout_cls(*args)  # type: ignore[return-value]
    return _resolve_type(ref.name, ns).__layout__  # type: ignore[return-value]


def _make_layout_fn(
    ref: TypeRef, param_map: dict[str, ParamDecl], ns: dict[str, object]
) -> Callable[[object], object]:
    """Return a closure ``(self) -> Layout | scalar`` for a parametric TypeRef."""
    if ref.name in param_map:
        p = param_map[ref.name]
        pname = ref.name
        if p.type.name == "Type":
            return lambda self, _n=pname: (
                dgen.type.type_constant(getattr(self, _n)).__layout__
            )
        return lambda self, _n=pname: getattr(self, _n).__constant__.to_json()
    layout_cls = _CONSTRUCTOR_LAYOUTS.get(ref.name)
    if layout_cls is not None:
        sub_fns = [_make_layout_fn(a, param_map, ns) for a in ref.args]
        return lambda self, _cls=layout_cls, _fns=sub_fns: _cls(
            *[f(self) for f in _fns]
        )
    # Type name — capture its layout at build time (static reference)
    static = _compute_static_layout(TypeRef(ref.name), ns)
    return lambda self, _l=static: _l


def _make_layout(
    td: TypeDecl, ns: dict[str, object]
) -> layout.Layout | property | None:
    """Compute the ``__layout__`` for a type: static Layout, property, or None."""
    param_map = {p.name: p for p in td.params}

    if td.layout is not None:
        primitive = _PRIMITIVE_LAYOUTS.get(td.layout)
        if primitive is not None:
            return primitive
        layout_cls = _CONSTRUCTOR_LAYOUTS[td.layout]
        if not td.params:
            return layout_cls(layout.Void())  # type: ignore[return-value]
        # Parametric layout keyword → property
        list_type_param = next(
            (
                p
                for p in td.params
                if p.type.name == "List"
                and p.type.args
                and p.type.args[0].name == "Type"
            ),
            None,
        )
        if td.layout == "Record" and list_type_param is not None:
            pname = list_type_param.name

            def _record_layout(self: object, _n: str = pname) -> layout.Layout:
                return layout.Record(
                    [
                        (str(i), dgen.type.type_constant(t).__layout__)
                        for i, t in enumerate(getattr(self, _n))
                    ]
                )

            return property(_record_layout)

        param_fns = []
        for p in td.params:
            pname = p.name
            if p.type.name == "Type":
                param_fns.append(
                    lambda self, _n=pname: (
                        dgen.type.type_constant(getattr(self, _n)).__layout__
                    )
                )
            else:
                param_fns.append(
                    lambda self, _n=pname: getattr(self, _n).__constant__.to_json()
                )

        def _param_layout(
            self: object, _cls: type = layout_cls, _fns: list = param_fns
        ) -> layout.Layout:
            return _cls(*[f(self) for f in _fns])  # type: ignore[return-value]

        return property(_param_layout)

    if td.data:
        is_parametric = any(_ref_has_params(df.type, param_map) for df in td.data)
        if not is_parametric:
            if len(td.data) == 1:
                return _compute_static_layout(td.data[0].type, ns)
            fields = [(df.name, _compute_static_layout(df.type, ns)) for df in td.data]
            return layout.Record(fields)  # type: ignore[return-value]
        # Parametric data → property
        if len(td.data) == 1:
            fn = _make_layout_fn(td.data[0].type, param_map, ns)
            return property(lambda self, _f=fn: _f(self))
        fns = [(df.name, _make_layout_fn(df.type, param_map, ns)) for df in td.data]

        def _data_layout(self: object, _fns: list = fns) -> layout.Layout:
            return layout.Record([(n, f(self)) for n, f in _fns])

        return property(_data_layout)

    return None


def _make_type_default(
    ref: TypeRef,
    type_map: dict[str, TypeDecl],
    known_names: set[str],
    ns: dict[str, object],
) -> object | None:
    """Build the default type value for an op's ``type`` field, or ``None``."""
    try:
        if not ref.args:
            if "." in ref.name:
                return None
            td = type_map.get(ref.name)
            if td is not None and any(p.default is None for p in td.params):
                return None
            if ref.name not in known_names:
                return None
            return _resolve_type(ref.name, ns)()
        td = type_map.get(ref.name)
        if td is None or len(ref.args) != len(td.params):
            return None
        kwargs: dict[str, object] = {}
        for arg, param in zip(ref.args, td.params):
            param_type = _resolve_type(param.type.name, ns)
            kwargs[param.name] = param_type().constant(_ast.literal_eval(arg.name))
        return _resolve_type(ref.name, ns)(**kwargs)
    except (ValueError, KeyError):
        return None


def _resolve_param_type(ref: TypeRef, ns: dict[str, object]) -> type:
    if ref.name == "Type":
        return dgen.TypeType
    return _resolve_type(ref.name, ns)


def _annotation_for_param(param: ParamDecl) -> str:
    if param.type.name == "List" and param.type.args:
        inner = param.type.args[0].name
        return f"list[Value[{('dgen.TypeType' if inner == 'Type' else inner)}]]"
    if param.type.name == "Type":
        return "Value[dgen.TypeType]"
    return f"Value[{param.type.name}]"


def _build_trait(td: TraitDecl, ns: dict[str, object]) -> type:
    trait_ns: dict[str, object] = {"__module__": ns.get("__name__", "")}
    annotations: dict[str, str] = {}
    for sf in td.statics:
        if sf.default is not None:
            trait_ns[sf.name] = _ast.literal_eval(sf.default)
        else:
            annotations[sf.name] = sf.type.name
    if annotations:
        trait_ns["__annotations__"] = annotations
    return type(td.name, (), trait_ns)


def _build_type(td: TypeDecl, dialect: Dialect, ns: dict[str, object]) -> type:
    type_ns: dict[str, object] = {"__module__": ns.get("__name__", "")}
    annotations: dict[str, str] = {}

    layout_val = _make_layout(td, ns)
    if layout_val is not None:
        type_ns["__layout__"] = layout_val

    for p in td.params:
        annotations[p.name] = _annotation_for_param(p)
        if p.default is not None:
            type_ns[p.name] = _resolve_type(p.default, ns)()

    if td.params:
        type_ns["__params__"] = tuple(
            (p.name, _resolve_param_type(p.type, ns)) for p in td.params
        )

    for sf in td.statics:
        if sf.default is not None:
            type_ns[sf.name] = _ast.literal_eval(sf.default)
        else:
            annotations[sf.name] = sf.type.name

    if annotations:
        type_ns["__annotations__"] = annotations

    bases: tuple[type, ...] = tuple(_resolve_type(t, ns) for t in td.traits) + (Type,)
    cls = dataclasses.dataclass(frozen=True)(type(td.name, bases, type_ns))
    dialect.type(td.name)(cls)
    return cls


def _build_op(
    od: OpDecl,
    dialect: Dialect,
    ns: dict[str, object],
    type_map: dict[str, TypeDecl],
    known_names: set[str],
) -> type:
    op_ns: dict[str, object] = {"__module__": ns.get("__name__", "")}
    annotations: dict[str, str] = {}

    for p in od.params:
        annotations[p.name] = _annotation_for_param(p)

    for op in od.operands:
        if op.default:
            annotations[op.name] = f"Value | {op.default}"
            op_ns[op.name] = _resolve_type(op.default, ns)()
        else:
            annotations[op.name] = "Value"

    annotations["type"] = "Type"
    ret = od.return_type
    if ret is not None and ret.name != "Type":
        default = _make_type_default(ret, type_map, known_names, ns)
        if default is not None:
            op_ns["type"] = default

    for block_name in od.blocks:
        annotations[block_name] = "Block"

    if od.params:
        op_ns["__params__"] = tuple(
            (p.name, _resolve_param_type(p.type, ns)) for p in od.params
        )

    if od.operands:
        op_ns["__operands__"] = tuple(
            (
                op.name,
                _resolve_type(op.type.name, ns) if op.type is not None else Type,
            )
            for op in od.operands
        )

    if od.blocks:
        op_ns["__blocks__"] = tuple(od.blocks)

    if od.constraints:
        op_ns["__constraints__"] = tuple(
            f"{c.lhs} ~= {c.pattern}"
            if c.kind == "match"
            else f"{c.lhs} == {c.rhs}"
            if c.kind == "eq"
            else c.expr
            for c in od.constraints
        )

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

    # Bootstrap: inject dgen, layout, Op, Type, Value, Block
    ns.setdefault("dgen", dgen)
    ns.setdefault("layout", layout)
    ns.setdefault("Type", Type)
    ns.setdefault("Value", Value)
    ns.setdefault("Op", Op)
    ns.setdefault("Block", Block)

    # Create dialect
    d = Dialect(dialect_name)
    ns[dialect_name] = d

    # Resolve imports
    effective_map = {"builtin": "dgen.dialects.builtin", **import_map}
    for imp in ast.imports:
        py_mod = effective_map.get(imp.module)
        if py_mod is None:
            continue
        mod = importlib.import_module(py_mod)
        if imp.names:
            for name in imp.names:
                ns[name] = getattr(mod, name)
        else:
            ns[imp.module] = mod

    # Build traits
    for td in ast.traits:
        ns[td.name] = _build_trait(td, ns)

    # Build types
    type_map = {td.name: td for td in ast.types}
    known_names: set[str] = set(type_map)
    for imp in ast.imports:
        known_names.update(imp.names)
    for td in ast.types:
        ns[td.name] = _build_type(td, d, ns)

    # Build ops
    for od in ast.ops:
        ns[_op_class_name(od.name)] = _build_op(od, d, ns, type_map, known_names)
