"""Runtime module builder for .dgen dialect specifications.

Creates Python classes directly via metaprogramming instead of generating and
exec()-ing source code.  Used by the dialect import hook and by direct calls
to :func:`dgen.imports.load`.
"""

from __future__ import annotations

import ast as _ast
import dataclasses
from collections.abc import Callable
from pathlib import Path

import dgen
from dgen import Block, Dialect, Op, Type, TypeType, Value, layout
from dgen.ir.constraints import TraitConstraint
from dgen.trait import Trait
from dgen.type import py_attr_name
from dgen.spec.ast import (
    DgenFile,
    HasTraitConstraint,
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
    "Some": layout.Some(),
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
    return _resolve_type(ref.name, ns)  # type: ignore[return-value]


def _is_type_kinded(param_type: type[Type]) -> bool:
    """True when *param_type* values are themselves types (metatype params)."""
    return issubclass(param_type, TypeType)


def _is_span_of_types(param_type: type[Type]) -> bool:
    """True when *param_type* is a Span whose element is type-kinded."""
    params: tuple[tuple[str, type[Type]], ...] = getattr(param_type, "__params__", ())
    if not any(_is_type_kinded(pt) for _, pt in params):
        return False
    try:
        kwargs = {name: TypeType() for name, pt in params if _is_type_kinded(pt)}
        return isinstance(param_type(**kwargs).__layout__, layout.Span)
    except (TypeError, KeyError):
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
                dgen.type.constant(getattr(self, _n)).__layout__
            )
        return lambda self, _n=pname: getattr(self, _n).__constant__.to_json()
    if ctor := _CONSTRUCTOR_LAYOUTS.get(ref.name):
        sub_fns = [_layout_fn(a, resolved_params, ns) for a in ref.args]
        return lambda self, _c=ctor, _fs=sub_fns: _c(*[f(self) for f in _fs])
    resolved_type = _resolve_type(ref.name, ns)
    static: object = _PRIMITIVE_LAYOUTS.get(ref.name) or resolved_type.__layout__
    if isinstance(static, property) and ref.args:
        # Parametric type used as a field (e.g. NDBuffer<shape, dtype>).
        # Build a closure that constructs the type with resolved args
        # and reads its layout at instance time.
        arg_fns = [_layout_fn(a, resolved_params, ns) for a in ref.args]
        param_names = [name for name, _ in resolved_type.__params__]

        def _resolve_layout(
            self: object,
            _cls: type[Type] = resolved_type,
            _pnames: list[str] = param_names,
            _afns: list[_LayoutFn] = arg_fns,
        ) -> layout.Layout:
            kwargs = {}
            for pname, afn in zip(_pnames, _afns):
                val = afn(self)
                if isinstance(val, layout.Layout):
                    # Layout → need to find the original type value, not the layout.
                    # Pass the attribute from self directly.
                    kwargs[pname] = getattr(self, pname)
                else:
                    kwargs[pname] = val
            return _cls(**kwargs).__layout__  # type: ignore[return-value]

        return _resolve_layout
    return lambda self, _l=static: _l


def _as_layout_or_property(fn: _LayoutFn, parametric: bool) -> layout.Layout | property:
    """Evaluate *fn* statically (``fn(None)``) or wrap as a property."""
    return property(fn) if parametric else fn(None)  # type: ignore[return-value]


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
                (p for p in td.params if _is_span_of_types(resolved_params[p.name])),
                None,
            )
        ):
            pname = lp.name
            # ``constant(...)`` resolves the parameter to its rich Python
            # form (a list of Type instances) regardless of whether the
            # underlying value is a ``PackOp`` (still iterable) or a
            # ``Constant`` (not iterable as-is).
            return property(
                lambda self, _n=pname: layout.Record(
                    [
                        (str(i), dgen.type.constant(t).__layout__)
                        for i, t in enumerate(dgen.type.constant(getattr(self, _n)))
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


def _trait_fn(
    ref: TypeRef,
    ns: dict[str, object],
    instance_params: set[str],
) -> Callable[[object], object]:
    """Return a ``(self) -> Type instance`` closure for a trait reference.

    - ``Numeric``                → ``lambda self: Numeric()``
    - ``Handler<Raise<E>>``      → ``lambda self: Handler(effect_type=Raise(error_type=self.E))``
      (closure reads the declaring class's ``E`` attribute at call time)
    - Bare instance-param name like ``error_type`` → ``lambda self: self.error_type``

    No ``_DeferredTrait`` / ``_InstanceParam`` wrappers needed — the closure
    itself handles deferral, and a purely-static closure just ignores self.
    """
    if ref.name in instance_params and not ref.args:
        pname = ref.name
        return lambda self, _n=pname: getattr(self, _n)
    cls = _resolve_type(ref.name, ns)
    if not ref.args:
        return lambda self, _c=cls: _c()
    param_names = [p_name for p_name, _ in getattr(cls, "__params__", ())]
    if len(param_names) != len(ref.args):
        raise TypeError(
            f"{ref.name}<...> expects {len(param_names)} args, got {len(ref.args)}"
        )
    arg_fns = [_trait_fn(a, ns, instance_params) for a in ref.args]
    return lambda self, _c=cls, _pn=param_names, _af=arg_fns: _c(
        **{pn: fn(self) for pn, fn in zip(_pn, _af)}
    )


def _install_traits_property(
    cls: type,
    trefs: list[TypeRef],
    ns: dict[str, object],
    instance_params: set[str],
) -> None:
    """Install a ``traits`` property on *cls* that yields this class's declared
    traits and chains to ``super().traits``.

    Has to run after the class object exists so the closure can bind it as
    the explicit ``super()`` reference: a generated method has no class-body
    scope for the implicit ``super()`` to read ``__class__`` from.
    """
    if not trefs:
        return
    fns = [_trait_fn(tref, ns, instance_params) for tref in trefs]

    def traits(self: object) -> object:
        for fn in fns:
            yield fn(self)
        yield from super(cls, self).traits

    cls.traits = property(traits)  # type: ignore[attr-defined]


def _compile_constraint(
    constraint: object,
    ns: dict[str, object],
    instance_params: set[str],
) -> object:
    """Compile a spec-AST constraint into its IR-level form.

    For now only ``HasTraitConstraint`` has an IR-level form (it's the only
    constraint that ``verify_constraints`` actually checks today). Other
    constraint kinds pass through unchanged.
    """
    if isinstance(constraint, HasTraitConstraint):
        build_target = _trait_fn(constraint.trait, ns, instance_params)
        return TraitConstraint(
            subject=constraint.lhs,
            build_target=build_target,  # type: ignore[arg-type]
        )
    return constraint


def _build_type(
    td: TypeDecl | TraitDecl,
    dialect: Dialect,
    ns: dict[str, object],
    *,
    base: type = Type,
    default_layout: layout.Layout | None = None,
) -> type:
    """Build a Type or Trait class.

    Traits are types: same params/statics machinery, same ``traits``
    property installation. The differences are local — base class (``Type``
    or ``Trait``) and layout default — passed in by the caller.
    ``TraitDecl`` carries no ``data``/``layout``/``traits``/``constraints``;
    those branches noop for traits via the AST attributes that aren't
    populated.
    """
    resolved_params = {p.name: _resolve_param_type(p.type, ns) for p in td.params}

    type_ns: dict[str, object] = {"__module__": ns.get("__name__", "")}
    annotations: dict[str, str] = {}

    layout_val = (
        _make_layout(td, ns) if isinstance(td, TypeDecl) else None
    ) or default_layout
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
    constraints = getattr(td, "constraints", ())
    if constraints:
        instance_params_for_constraints = {p.name for p in td.params}
        type_ns["__constraints__"] = tuple(
            _compile_constraint(c, ns, instance_params_for_constraints)
            for c in constraints
        )

    if annotations:
        type_ns["__annotations__"] = annotations

    cls = dataclasses.dataclass(eq=False)(type(td.name, (base,), type_ns))
    # Traits are NEVER base classes — ``has trait T<...>`` installs a
    # ``traits`` property that yields T(...) and chains to ``super().traits``.
    instance_params = {p.name for p in td.params}
    type_traits = getattr(td, "traits", ())
    _install_traits_property(cls, type_traits, ns, instance_params)
    dialect.type(td.name)(cls)
    return cls


def _build_trait(td: TraitDecl, dialect: Dialect, ns: dict[str, object]) -> type:
    """Trait declaration: a Type with ``layout Void`` by default."""
    return _build_type(td, dialect, ns, base=Trait, default_layout=layout.Void())


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

    # Blocks come before `type` so that any default value on `type` is a
    # trailing default — required now that we no longer emit kw_only=True.
    annotations.update({py_attr_name(block_name): "Block" for block_name in od.blocks})

    annotations["type"] = "Type"
    ret = od.return_type
    if ret is not None:
        if ret.name == "Type":
            op_ns["type"] = TypeType()
        else:
            default = _make_type_default(ret, type_map, known_names, ns)
            if default is not None:
                op_ns["type"] = default

    if od.params:
        op_ns["__params__"] = tuple(
            (p.name, resolved_params[p.name]) for p in od.params
        )
    if od.operands:
        param_name_set = {p.name for p in od.params}
        op_ns["__operands__"] = tuple(
            (
                op.name,
                # Operand annotated with an op-parameter name (e.g. ``error:
                # error_type``): the static expected type is the generic
                # ``Type`` — the concrete type is refined per-instance from
                # the op's actual parameter value.
                Type
                if op.type is not None and op.type.name in param_name_set
                else _resolve_type(op.type.name, ns)
                if op.type is not None
                else Value,
            )
            for op in od.operands
        )
    if od.blocks:
        op_ns["__blocks__"] = tuple(od.blocks)
    if od.constraints:
        op_ns["__constraints__"] = tuple(
            _compile_constraint(c, ns, {p.name for p in od.params})
            for c in od.constraints
        )

    op_ns["__annotations__"] = annotations
    cls = dataclasses.dataclass(eq=False)(type(_op_class_name(od.name), (Op,), op_ns))
    # An op can declare a trait it "is" (``op add_nums: has trait Numeric``);
    # ``has_trait`` on the op value finds it via the same property chain.
    instance_params = {p.name for p in od.params}
    _install_traits_property(cls, od.traits, ns, instance_params)
    dialect.op(od.name)(cls)
    return cls


def build(
    ast: DgenFile,
    qualname: str,
    dgen_dir: Path,
    ns: dict[str, object] | None = None,
) -> Dialect:
    """Build a Dialect from an AST.

    ``ns`` is the namespace used to resolve names during construction
    (imported types, locally-defined types, dgen builtins).  When called
    from the Python import hook ``ns`` is the loading module's ``__dict__``;
    otherwise pass ``None`` for an internal scratch dict.
    """
    if ns is None:
        ns = {}

    ns.setdefault("dgen", dgen)
    ns.setdefault("layout", layout)
    # ``Type`` in dgen source is the metatype: any value of type ``Type`` is
    # itself a type, with ``TypeValue`` layout. Bind the dgen name to
    # ``TypeType`` so all resolution paths (params, fields, return types,
    # static fallbacks in ``_layout_fn``) reach a class that has ``__layout__``.
    ns.setdefault("Type", TypeType)
    ns.setdefault("Value", Value)
    ns.setdefault("Op", Op)
    ns.setdefault("Block", Block)

    d = Dialect(qualname)
    ns[qualname.split(".")[-1]] = d

    for imp in ast.imports:
        if imp.relative:
            for name in imp.names:
                ns[name] = dgen.imports.load(name, _from=dgen_dir)
        else:
            other = dgen.imports.load(imp.module)
            if imp.names:
                ns.update({n: getattr(other, n) for n in imp.names})
            else:
                ns[imp.module.split(".")[-1]] = other

    for td in ast.traits:
        ns[td.name] = _build_trait(td, d, ns)

    type_map = {td.name: td for td in ast.types}
    known_names = set(type_map) | {name for imp in ast.imports for name in imp.names}
    for td in ast.types:
        ns[td.name] = _build_type(td, d, ns)

    for od in ast.ops:
        ns[_op_class_name(od.name)] = _build_op(od, d, ns, type_map, known_names)

    return d
