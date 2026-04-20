"""Python .pyi stub generator by introspecting live dialect modules.

Instead of generating stubs from the AST, this module inspects the classes
that ``build.py`` already created at import time.  The import hook loads the
``.dgen`` file, ``build.py`` materialises real dataclass types, and this
module walks those live objects to emit ``.pyi`` type stubs.
"""

from __future__ import annotations

import dataclasses
import re
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
    if isinstance(value, TypeType):
        return "dgen.TypeType()"
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


def _trait_stub(name: str, cls: type, type_to_name: dict[type, str]) -> Iterator[str]:
    """Yield stub lines for a trait class.

    Unparameterized traits are simple class aliases of ``Trait``. Parametric
    traits (with ``__params__``) emit as dataclasses — same shape as
    parametric types — so they can be instantiated in user code and compared
    structurally.
    """
    param_types = dict(getattr(cls, "__params__", ()))
    if not param_types:
        yield f"class {name}(Trait):"
        yield "    ..."
        yield ""
        return
    yield "@dataclass(eq=False)"
    yield f"class {name}(Trait):"
    for fname, pt in param_types.items():
        yield f"    {fname}: {_param_annotation(pt, type_to_name)}"
    yield ""


def _stub_class(
    cls: type,
    type_to_name: dict[type, str],
) -> Iterator[str]:
    """Yield stub lines for one type or op dataclass."""
    yield "@dataclass(eq=False)"

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
    needs_trait = any(Trait in cls.__bases__ for cls in dialect.types.values())
    has_ops = bool(dialect.ops)
    has_non_trait_types = any(
        Trait not in cls.__bases__ for cls in dialect.types.values()
    )
    has_value_fields = any(
        any("Value" in f.type for f in _own_fields(cls))
        for cls in (*dialect.types.values(), *dialect.ops.values())
        if dataclasses.is_dataclass(cls)
    )

    dgen_imports = sorted(
        {"Dialect"}
        | ({"Op"} if has_ops else set())
        | ({"Type"} if has_non_trait_types or has_ops else set())
        | ({"Value"} if has_value_fields else set())
        | ({"Block"} if needs_block else set())
        | ({"Trait"} if needs_trait else set())
    )

    # Emit class stubs first; the imports-needed set is derived from what
    # actually appears in the body, so unused `.dgen`-level imports don't leak
    # into the stub.
    body_lines: list[str] = []
    for name, cls in dialect.types.items():
        # A trait DECLARATION (vs a type that "has trait X") has Trait as a
        # direct base. Types that just implement a trait go through the
        # normal class-stub path so their bases reflect the trait's class.
        if Trait in cls.__bases__:
            body_lines.extend(_trait_stub(name, cls, type_to_name))
        else:
            body_lines.extend(_stub_class(cls, type_to_name))
    for cls in dialect.ops.values():
        body_lines.extend(_stub_class(cls, type_to_name))

    body_text = "\n".join(body_lines)
    used_imports = {
        src: sorted({n for n in names if _name_used(n, body_text)})
        for src, names in imported_type_names.items()
    }
    used_imports = {src: names for src, names in used_imports.items() if names}

    lines: list[str] = [
        f"# GENERATED by dgen from {dialect_name}.dgen — do not edit.",
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
    ]
    if _name_used("dgen.", body_text):
        lines.append("import dgen")
    lines.append(f"from dgen import {', '.join(dgen_imports)}")

    for alias, mod in sorted(imported_mods.items()):
        lines.append(f"import {mod.__name__} as {alias}")
    for src, names in sorted(used_imports.items()):
        lines.append(f"from {src} import {', '.join(names)}")

    lines += ["", f'{dialect_name} = Dialect("{dialect_name}")', ""]
    lines.append(body_text)

    return "\n".join(lines) + "\n"


def _name_used(name: str, body: str) -> bool:
    """Whole-word match for *name* in the emitted body text.

    Accepts a trailing ``.`` (e.g. ``"dgen."``) for namespaced lookups.
    """
    if name.endswith("."):
        return re.search(rf"\b{re.escape(name)}", body) is not None
    return re.search(rf"\b{re.escape(name)}\b", body) is not None
