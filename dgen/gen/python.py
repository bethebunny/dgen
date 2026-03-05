"""Python code generator for .dgen dialect specifications."""

from __future__ import annotations

from dgen.gen.ast import (
    DataField,
    DgenFile,
    OpDecl,
    OperandDecl,
    ParamDecl,
    TraitDecl,
    TypeDecl,
    TypeRef,
)

# ===----------------------------------------------------------------------=== #
# Name mapping: .dgen ASM name -> Python class name
# ===----------------------------------------------------------------------=== #

# Names that keep their .dgen name as-is (no "Type" suffix).
# Includes both types (Nil, String, etc.) and traits (HasSingleBlock).
_NAME_KEEP: frozenset[str] = frozenset(
    {
        "Nil",
        "String",
        "List",
        "InferredShapeTensor",
        "HasSingleBlock",
    }
)

# Known trait names (subset of _NAME_KEEP that are traits, not types)
_KNOWN_TRAITS: frozenset[str] = frozenset(
    {
        "HasSingleBlock",
    }
)

# Map from .dgen type names to Python layout constructor calls.
# These are the leaf types whose layout is intrinsic.
_TYPE_TO_LAYOUT: dict[str, str] = {
    "Index": "Int()",
    "index": "Int()",
    "F64": "Float64()",
    "f64": "Float64()",
    "Nil": "Void()",
    "Byte": "Byte()",
    "StringLayout": "StringLayout(Byte())",
}

# Map from .dgen compound type names to Python layout constructor names.
_COMPOUND_TO_LAYOUT: dict[str, str] = {
    "Array": "Array",
    "Pointer": "Pointer",
    "FatPointer": "FatPointer",
}

# Layout classes that need to be imported from dgen.layout
_LAYOUT_CONSTRUCTORS: frozenset[str] = frozenset(
    {"Array", "Pointer", "FatPointer", "Void", "Byte", "Int", "Float64", "StringLayout"}
)


def _type_class_name(asm_name: str) -> str:
    """Derive Python class name from ASM type name."""
    if asm_name in _NAME_KEEP:
        return asm_name
    if asm_name == "Type":
        return "Type"
    # Handle lowercase names like "index", "f64"
    if asm_name[0].islower() or asm_name[0].isdigit():
        # "index" -> "Index" -> "IndexType"
        # "f64" -> "F64" -> "F64Type"
        camel = asm_name[0].upper() + asm_name[1:]
        return camel + "Type"
    # CamelCase names: add "Type" suffix
    return asm_name + "Type"


def _op_class_name(asm_name: str) -> str:
    """Derive Python class name from ASM op name."""
    parts = asm_name.split("_")
    camel = "".join(p.capitalize() for p in parts)
    return camel + "Op"


def _resolve_type_ref(ref: TypeRef) -> str:
    """Resolve a TypeRef to a Python class name for use in __params__/__operands__."""
    if ref.name == "Type":
        return "Type"
    if ref.name == "list":
        if ref.args:
            return _resolve_type_ref(ref.args[0])
        return "Type"
    return _type_class_name(ref.name)


def _annotation_for_param(param: ParamDecl) -> str:
    """Generate Python type annotation for a compile-time parameter."""
    if param.type.name == "list":
        inner = _resolve_type_ref(param.type.args[0]) if param.type.args else "Type"
        return f"list[Value[{inner}]]"
    if param.type.name == "Type":
        return "Type"
    return f"Value[{_type_class_name(param.type.name)}]"


def _annotation_for_operand(operand: OperandDecl) -> str:
    """Generate Python type annotation for a runtime operand."""
    if operand.type.name == "list":
        return "list[Value]"
    return "Value"


# ===----------------------------------------------------------------------=== #
# Code generation
# ===----------------------------------------------------------------------=== #


def generate(
    ast: DgenFile,
    dialect_name: str,
    import_map: dict[str, str] | None = None,
) -> str:
    """Generate Python source code from a DgenFile AST.

    Args:
        ast: The parsed .dgen file.
        dialect_name: Name for the Dialect("name") instance.
        import_map: Maps .dgen module names to Python module paths.
            E.g. {"builtin": "dgen.dialects.builtin", "affine": "toy.dialects.affine"}
    """
    gen = _Generator(ast, dialect_name, import_map or {})
    return gen.emit()


class _Generator:
    def __init__(
        self,
        ast: DgenFile,
        dialect_name: str,
        import_map: dict[str, str],
    ) -> None:
        self.ast = ast
        self.dialect_name = dialect_name
        self.import_map = import_map
        self._lines: list[str] = []

        # Pre-compute: which layout imports are needed
        self._layout_classes: set[str] = set()
        self._needs_block = False
        self._trait_names: set[str] = {t.name for t in ast.traits}
        # Also track imported trait names
        for imp in ast.imports:
            for name in imp.names:
                if name in _KNOWN_TRAITS:
                    self._trait_names.add(name)

        # Scan for required layout imports from data field type references
        for td in ast.types:
            if td.data:
                self._collect_layout_imports_from_type(td.data.type, td.params)
        # Scan for Block usage
        for od in ast.ops:
            if od.blocks:
                self._needs_block = True

    def _collect_layout_imports_from_type(
        self, type_ref: TypeRef, params: list[ParamDecl]
    ) -> None:
        """Collect layout imports needed to resolve a data field type reference."""
        param_names = {p.name for p in params}
        name = type_ref.name
        # Check if this type name resolves to a layout
        if name in _TYPE_TO_LAYOUT:
            # Extract class names from constructor call expressions
            expr = _TYPE_TO_LAYOUT[name]
            for cls_name in _LAYOUT_CONSTRUCTORS:
                if cls_name in expr:
                    self._layout_classes.add(cls_name)
        elif name in _COMPOUND_TO_LAYOUT:
            layout_name = _COMPOUND_TO_LAYOUT[name]
            if layout_name in _LAYOUT_CONSTRUCTORS:
                self._layout_classes.add(layout_name)
        # Recurse into type args (but skip param references)
        for arg in type_ref.args:
            if arg.name not in param_names:
                self._collect_layout_imports_from_type(arg, params)

    def emit(self) -> str:
        self._emit_header()
        self._emit_imports()
        self._emit_dialect()
        for trait in self.ast.traits:
            self._emit_trait(trait)
        for td in self.ast.types:
            self._emit_type(td)
        for od in self.ast.ops:
            self._emit_op(od)
        return "\n".join(self._lines) + "\n"

    def _line(self, text: str = "") -> None:
        self._lines.append(text)

    def _emit_header(self) -> None:
        self._line(f"# GENERATED by dgen from {self.dialect_name}.dgen — do not edit.")
        self._line()
        self._line("from __future__ import annotations")
        self._line()

    def _emit_imports(self) -> None:
        self._line("from dataclasses import dataclass")
        self._line()

        # dgen framework imports
        dgen_names = ["Dialect", "Op", "Type", "Value"]
        if self._needs_block:
            dgen_names.insert(0, "Block")
        self._line(f"from dgen import {', '.join(sorted(dgen_names))}")

        # Layout imports
        layout_names: list[str] = sorted(self._layout_classes)
        if layout_names:
            self._line(f"from dgen.layout import {', '.join(layout_names)}")

        # Cross-dialect imports
        for imp in self.ast.imports:
            python_module = self.import_map.get(imp.module)
            if python_module:
                py_names = [_type_class_name(n) for n in imp.names]
                self._line(f"from {python_module} import {', '.join(py_names)}")

        self._line()

    def _emit_dialect(self) -> None:
        self._line(f'{self.dialect_name} = Dialect("{self.dialect_name}")')
        self._line()

    def _emit_trait(self, trait: TraitDecl) -> None:
        self._line()
        self._line(f"class {trait.name}:")
        self._line("    pass")
        self._line()

    def _emit_type(self, td: TypeDecl) -> None:
        cls_name = _type_class_name(td.name)
        is_parametric = td.data is not None and self._data_is_parametric(
            td.data, td.params
        )

        self._line()
        self._line(f'@{self.dialect_name}.type("{td.name}")')
        self._line("@dataclass(frozen=True)")
        self._line(f"class {cls_name}(Type):")

        body_lines: list[str] = []

        # Static layout (not parametric)
        if td.data and not is_parametric:
            body_lines.append(
                f"    __layout__ = {self._resolve_data_static(td.data.type)}"
            )

        # Parameters
        for p in td.params:
            ann = _annotation_for_param(p)
            if p.default:
                default_cls = _type_class_name(p.default)
                body_lines.append(f"    {p.name}: {ann} = {default_cls}()")  # type: ignore[unreachable]
            else:
                body_lines.append(f"    {p.name}: {ann}")

        # __params__ tuple
        if td.params:
            parts = [f'("{p.name}", {_resolve_type_ref(p.type)})' for p in td.params]
            body_lines.append(f"    __params__ = ({', '.join(parts)},)")

        # Parametric layout property
        if td.data and is_parametric:
            body_lines.append("")
            body_lines.append("    @property")
            return_type = self._data_return_type(td.data.type)
            body_lines.append(f"    def __layout__(self) -> {return_type}:")
            body_lines.append(
                f"        return {self._resolve_data_parametric(td.data.type, td.params)}"
            )

        if not body_lines:
            body_lines.append("    pass")

        for line in body_lines:
            self._line(line)
        self._line()

    def _emit_op(self, od: OpDecl) -> None:
        cls_name = _op_class_name(od.name)
        has_trait = bool(od.blocks) and "HasSingleBlock" in self._trait_names

        self._line()
        self._line(f'@{self.dialect_name}.op("{od.name}")')
        self._line("@dataclass(eq=False, kw_only=True)")

        bases = "Op"
        if has_trait:
            bases = "HasSingleBlock, Op"
        self._line(f"class {cls_name}({bases}):")

        body_lines: list[str] = []

        # Parameters (compile-time, before operands)
        for p in od.params:
            ann = _annotation_for_param(p)
            body_lines.append(f"    {p.name}: {ann}")

        # Operands (runtime)
        for op in od.operands:
            ann = _annotation_for_operand(op)
            if op.default:
                default_cls = _type_class_name(op.default)
                body_lines.append(
                    f"    {op.name}: {ann} | {default_cls} = {default_cls}()"
                )
            else:
                body_lines.append(f"    {op.name}: {ann}")

        # Return type field
        ret = od.return_type
        if ret.name == "Type":
            body_lines.append("    type: Type")
        else:
            default_expr = self._type_default_expr(ret)
            if default_expr is not None:
                body_lines.append(f"    type: Type = {default_expr}")
            else:
                body_lines.append("    type: Type")

        # Block fields
        for block_name in od.blocks:
            body_lines.append(f"    {block_name}: Block")

        # __params__ tuple
        if od.params:
            parts = [f'("{p.name}", {_resolve_type_ref(p.type)})' for p in od.params]
            body_lines.append(f"    __params__ = ({', '.join(parts)},)")

        # __operands__ tuple
        if od.operands:
            parts = [
                f'("{op.name}", {_resolve_type_ref(op.type)})' for op in od.operands
            ]
            body_lines.append(f"    __operands__ = ({', '.join(parts)},)")

        # __blocks__ tuple
        if od.blocks:
            parts = [f'"{b}"' for b in od.blocks]
            body_lines.append(f"    __blocks__ = ({', '.join(parts)},)")

        for line in body_lines:
            self._line(line)
        self._line()

    # ===----------------------------------------------------------------------=== #
    # Data field -> layout resolution
    # ===----------------------------------------------------------------------=== #

    def _data_is_parametric(self, data: DataField, params: list[ParamDecl]) -> bool:
        """Check if a data field type references any type parameters."""
        param_names = {p.name for p in params}
        return self._type_ref_has_params(data.type, param_names)

    def _type_ref_has_params(self, ref: TypeRef, param_names: set[str]) -> bool:
        if ref.name in param_names:
            return True
        return any(self._type_ref_has_params(arg, param_names) for arg in ref.args)

    def _resolve_data_static(self, ref: TypeRef) -> str:
        """Resolve a non-parametric data field type to a static layout expression."""
        if ref.name in _TYPE_TO_LAYOUT:
            return _TYPE_TO_LAYOUT[ref.name]
        if ref.name in _COMPOUND_TO_LAYOUT:
            constructor = _COMPOUND_TO_LAYOUT[ref.name]
            args = ", ".join(self._resolve_data_static(arg) for arg in ref.args)
            return f"{constructor}({args})"
        return ref.name

    def _resolve_data_parametric(self, ref: TypeRef, params: list[ParamDecl]) -> str:
        """Resolve a parametric data field type to a layout expression for a @property."""
        param_map = {p.name: p for p in params}
        return self._resolve_ref_parametric(ref, param_map)

    def _resolve_ref_parametric(
        self, ref: TypeRef, param_map: dict[str, ParamDecl]
    ) -> str:
        if ref.name in param_map:
            p = param_map[ref.name]
            if p.type.name == "Type":
                return f"self.{ref.name}.__layout__"
            return f"self.{ref.name}.__constant__.to_json()"
        if ref.name in _TYPE_TO_LAYOUT:
            return _TYPE_TO_LAYOUT[ref.name]
        if ref.name in _COMPOUND_TO_LAYOUT:
            constructor = _COMPOUND_TO_LAYOUT[ref.name]
            args = ", ".join(
                self._resolve_ref_parametric(arg, param_map) for arg in ref.args
            )
            return f"{constructor}({args})"
        return ref.name

    def _data_return_type(self, ref: TypeRef) -> str:
        """Determine the return type annotation for a layout property."""
        if ref.name in _COMPOUND_TO_LAYOUT:
            return _COMPOUND_TO_LAYOUT[ref.name]
        return "Layout"

    def _type_default_expr(self, ref: TypeRef) -> str | None:
        """Generate a default construction expression for a return type.

        Handles both simple types (Nil -> Nil()) and parameterized types
        with literal args (Int<64> -> IntType(bits=IndexType().constant(64))).
        """
        cls_name = _type_class_name(ref.name)
        if not ref.args:
            if self._type_has_no_required_params(ref):
                return f"{cls_name}()"
            return None
        # Find type declaration to get param names and types
        td = next((t for t in self.ast.types if t.name == ref.name), None)
        if td is None or len(ref.args) != len(td.params):
            return None
        parts = []
        for arg, param in zip(ref.args, td.params):
            param_type_cls = _type_class_name(param.type.name)
            parts.append(f"{param.name}={param_type_cls}().constant({arg.name})")
        return f"{cls_name}({', '.join(parts)})"

    def _type_has_no_required_params(self, ref: TypeRef) -> bool:
        """Check if a type can be constructed with no arguments."""
        no_arg = {"Nil", "Index", "index", "F64", "f64", "String"}
        if ref.name in no_arg:
            return True
        for td in self.ast.types:
            if td.name == ref.name:
                return len(td.params) == 0
        return False
