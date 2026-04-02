"""Python .pyi stub generator by introspecting live dialect modules.

Instead of generating stubs from the AST, this module inspects the classes
that ``build.py`` already created at import time.  The import hook loads the
``.dgen`` file, ``build.py`` materialises real dataclass types, and this
module walks those live objects to emit ``.pyi`` type stubs.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Iterator
from types import ModuleType

from dgen import Block, Dialect, Op, Trait, Type, TypeType, Value
from dgen.type import Constant

_DGEN_CORE: frozenset[type] = frozenset({Type, Op, Value, Block, TypeType, Trait})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_type_kinded(param_type: type[Type]) -> bool:
    return param_type is Type or issubclass(param_type, TypeType)


def _own_fields(cls: type) -> list[dataclasses.Field[object]]:
    """Dataclass fields defined on *cls* itself (not inherited from bases)."""
    if not dataclasses.is_dataclass(cls):
        return []
    own = getattr(cls, "__annotations__", {})
    return [f for f in dataclasses.fields(cls) if f.name in own]


def _stub_repr(value: object) -> str:
    """Format a value for a .pyi stub, with readable Constant representations."""
    if isinstance(value, Constant):
        data = value.value.unpack()
        scalar = data[0] if len(data) == 1 else data
        return f"{_stub_repr(value.type)}.constant({scalar!r})"
    if isinstance(value, Type) and dataclasses.is_dataclass(value):
        fields = _own_fields(type(value))
        if not fields:
            return f"{type(value).__name__}()"
        parts = [f"{f.name}={_stub_repr(getattr(value, f.name))}" for f in fields]
        return f"{type(value).__name__}({', '.join(parts)})"
    return repr(value)


def _param_annotation(param_type: type[Type], type_to_name: dict[type, str]) -> str:
    """Annotation string for a compile-time parameter field."""
    if _is_type_kinded(param_type):
        return "Value[dgen.TypeType]"
    return f"Value[{type_to_name.get(param_type, param_type.__name__)}]"


def _build_type_to_name(
    ns: dict[str, object],
    imported_mods: dict[str, ModuleType],
) -> dict[type, str]:
    """Build a reverse map from live type classes to their stub-qualified name."""
    result: dict[type, str] = {}
    for name, val in ns.items():
        if isinstance(val, type) and not name.startswith("_"):
            result[val] = name
    for alias, mod in imported_mods.items():
        for attr, val in vars(mod).items():
            if isinstance(val, type) and val not in result:
                result[val] = f"{alias}.{attr}"
    return result


# ---------------------------------------------------------------------------
# Class stub emitters
# ---------------------------------------------------------------------------


_TRAIT_INTERNAL: frozenset[str] = frozenset({"__module__", "asm_name", "dialect"})


def _trait_stub(name: str, cls: type) -> Iterator[str]:
    """Yield stub lines for a trait class."""
    annotations = typing.get_type_hints(cls)
    has_body = False
    yield f"class {name}(Trait):"
    for attr_name, ann in annotations.items():
        has_body = True
        attr_val = getattr(cls, attr_name, dataclasses.MISSING)
        if attr_val is not dataclasses.MISSING:
            yield f"    {attr_name} = {attr_val!r}"
        else:
            yield f"    {attr_name}: {ann}"
    for attr_name, attr_val in vars(cls).items():
        if (
            attr_name not in annotations
            and not attr_name.startswith("_")
            and attr_name not in _TRAIT_INTERNAL
        ):
            has_body = True
            yield f"    {attr_name} = {attr_val!r}"
    if not has_body:
        yield "    ..."
    yield ""


def _stub_class(
    cls: type,
    type_to_name: dict[type, str],
    *,
    frozen: bool,
) -> Iterator[str]:
    """Yield stub lines for one type or op dataclass."""
    if frozen:
        yield "@dataclass(frozen=True, eq=False)"
    else:
        yield "@dataclass(eq=False, kw_only=True)"

    bases = ", ".join(
        type_to_name.get(b, b.__name__) for b in cls.__bases__ if b is not object
    )
    yield f"class {cls.__name__}({bases}):"

    fields = {f.name: f for f in _own_fields(cls)}
    param_types = dict(getattr(cls, "__params__", ()))

    if fields:
        for name, f in fields.items():
            if name in param_types:
                ann = _param_annotation(param_types[name], type_to_name)
            else:
                ann = f.type
            if f.default is dataclasses.MISSING:
                yield f"    {name}: {ann}"
            else:
                yield f"    {name}: {ann} = {_stub_repr(f.default)}"
    else:
        yield "    ..."
    yield ""


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------


def generate_pyi(module: ModuleType, dialect_name: str) -> str:
    """Generate a ``.pyi`` type stub by introspecting a live dialect module."""
    ns = vars(module)
    dialect: Dialect = ns[dialect_name]
    current_module = module.__name__

    # Imported module aliases (e.g. affine → toy.dialects.affine)
    imported_mods: dict[str, ModuleType] = {
        name: val
        for name, val in ns.items()
        if isinstance(val, ModuleType)
        and not name.startswith("_")
        and name not in ("dgen", "layout")
    }

    type_to_name = _build_type_to_name(ns, imported_mods)

    # Specifically-imported names from other dialect modules
    imported_type_names: dict[str, list[str]] = {}
    for name, val in ns.items():
        if (
            isinstance(val, type)
            and not name.startswith("_")
            and val not in _DGEN_CORE
            and getattr(val, "__module__", "") != current_module
            # Only include names the dialect actually uses (types/traits, not dgen core)
            and not (
                issubclass(val, (Type, Op)) and getattr(val, "dialect", None) is dialect
            )
        ):
            src = getattr(val, "__module__", "")
            if src and src != current_module:
                imported_type_names.setdefault(src, []).append(name)

    # Detect if Block is needed in imports
    needs_block = any(
        any(f.type == "Block" for f in _own_fields(cls))
        for cls in dialect.ops.values()
        if dataclasses.is_dataclass(cls)
    )
    needs_trait = bool(dialect.traits)

    dgen_imports = sorted(
        {"Dialect", "Op", "Type", "Value"}
        | ({"Block"} if needs_block else set())
        | ({"Trait"} if needs_trait else set())
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
    ]

    for alias, mod in sorted(imported_mods.items()):
        lines.append(f"import {mod.__name__} as {alias}")
    for src, names in sorted(imported_type_names.items()):
        lines.append(f"from {src} import {', '.join(sorted(names))}")

    lines += ["", f'{dialect_name} = Dialect("{dialect_name}")', ""]

    for name, val in dialect.traits.items():
        lines.extend(_trait_stub(name, val))
    for cls in dialect.types.values():
        lines.extend(_stub_class(cls, type_to_name, frozen=True))
    for cls in dialect.ops.values():
        lines.extend(_stub_class(cls, type_to_name, frozen=False))

    return "\n".join(lines) + "\n"
