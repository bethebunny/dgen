"""Python code generator for .dgen dialect specifications."""

from __future__ import annotations

from dgen.gen.ast import (
    DgenFile,
    LayoutExpr,
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

# Types that keep their .dgen name as-is (no "Type" suffix)
_TYPE_NAME_KEEP: frozenset[str] = frozenset(
    {
        "Nil",
        "String",
        "List",
        "InferredShapeTensor",
    }
)

# Layout primitives that are imported as module-level singletons
_LAYOUT_SINGLETONS: frozenset[str] = frozenset(
    {
        "INT",
        "FLOAT64",
        "VOID",
        "BYTE",
    }
)

# Layout constructors that need to be imported
_LAYOUT_CONSTRUCTORS: frozenset[str] = frozenset(
    {
        "Array",
        "Pointer",
        "FatPointer",
        "Bytes",
    }
)


def _type_class_name(asm_name: str) -> str:
    """Derive Python class name from ASM type name."""
    if asm_name in _TYPE_NAME_KEEP:
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
    # "transpose" -> "Transpose" -> "TransposeOp"
    # "add_index" -> "AddIndex" -> "AddIndexOp"
    # "for" -> "For" -> "ForOp"
    parts = asm_name.split("_")
    camel = "".join(p.capitalize() for p in parts)
    return camel + "Op"


def _resolve_type_ref(ref: TypeRef) -> str:
    """Resolve a TypeRef to a Python class name for use in __params__/__operands__."""
    if ref.name == "Type":
        return "Type"
    if ref.name == "list":
        # list<String> -> String (the element type is what goes in the tuple)
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
    # Operands are always Value (no type parameter), per existing convention
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
        self._layout_singletons: set[str] = set()
        self._layout_constructors: set[str] = set()
        self._needs_block = False
        self._trait_names: set[str] = {t.name for t in ast.traits}

        # Scan for required layout imports
        for td in ast.types:
            if td.layout:
                self._collect_layout_imports(td.layout, td.params)
        # Scan for Block usage
        for od in ast.ops:
            if od.blocks:
                self._needs_block = True

    def _collect_layout_imports(
        self, layout: LayoutExpr, params: list[ParamDecl]
    ) -> None:
        if layout.name in _LAYOUT_SINGLETONS:
            self._layout_singletons.add(layout.name)
        elif layout.name in _LAYOUT_CONSTRUCTORS:
            self._layout_constructors.add(layout.name)
        for arg in layout.args:
            if arg in _LAYOUT_SINGLETONS:
                self._layout_singletons.add(arg)
            elif arg in _LAYOUT_CONSTRUCTORS:
                self._layout_constructors.add(arg)
            # param refs don't need layout imports

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
        layout_names: list[str] = sorted(self._layout_singletons) + sorted(
            self._layout_constructors
        )
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
        is_parametric_layout = td.layout is not None and self._layout_is_parametric(
            td.layout, td.params
        )

        self._line()
        self._line(f'@{self.dialect_name}.type("{td.name}")')
        if td.params:
            self._line("@dataclass(frozen=True)")
        else:
            self._line("@dataclass(frozen=True)")
        self._line(f"class {cls_name}(Type):")

        body_lines: list[str] = []

        # Static layout (not parametric)
        if td.layout and not is_parametric_layout:
            body_lines.append(
                f"    __layout__ = {self._format_layout_value(td.layout)}"
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
        if td.layout and is_parametric_layout:
            body_lines.append("")
            body_lines.append("    @property")
            return_type = self._layout_return_type(td.layout)
            body_lines.append(f"    def __layout__(self) -> {return_type}:")
            body_lines.append(
                f"        return {self._format_layout_property(td.layout, td.params)}"
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
            # Generic return — no default
            body_lines.append("    type: Type")
        else:
            # Concrete return — generate default if type has no required params
            ret_cls = _resolve_type_ref(ret)
            if self._type_has_no_required_params(ret):
                body_lines.append(f"    type: Type = {ret_cls}()")
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
    # Layout helpers
    # ===----------------------------------------------------------------------=== #

    def _layout_is_parametric(
        self, layout: LayoutExpr, params: list[ParamDecl]
    ) -> bool:
        """Check if a layout expression references any type parameters."""
        param_names = {p.name for p in params}
        return any(arg in param_names for arg in layout.args)

    def _format_layout_value(self, layout: LayoutExpr) -> str:
        """Format a static layout expression: INT, FatPointer(BYTE), etc."""
        if not layout.args:
            return layout.name
        args_str = ", ".join(layout.args)
        return f"{layout.name}({args_str})"

    def _format_layout_property(
        self, layout: LayoutExpr, params: list[ParamDecl]
    ) -> str:
        """Format a parametric layout expression for a @property body."""
        param_map = {p.name: p for p in params}
        args: list[str] = []
        for arg in layout.args:
            if arg in param_map:
                p = param_map[arg]
                if p.type.name == "Type":
                    # Type param: use .__layout__
                    args.append(f"self.{arg}.__layout__")
                else:
                    # Index/other param: extract constant value
                    args.append(f"self.{arg}.__constant__.unpack()[0]")
            else:
                # Layout primitive: use as-is
                args.append(arg)
        return f"{layout.name}({', '.join(args)})"

    def _layout_return_type(self, layout: LayoutExpr) -> str:
        """Determine the return type annotation for a layout property."""
        return layout.name if layout.name in _LAYOUT_CONSTRUCTORS else "Layout"

    def _type_has_no_required_params(self, ref: TypeRef) -> bool:
        """Check if a type can be constructed with no arguments.

        Returns True for simple types (Nil, IndexType, F64Type, etc.)
        and False for parameterized types (List, ShapeType, etc.).
        """
        # Known no-arg types
        no_arg = {"Nil", "Index", "index", "F64", "f64", "String"}
        if ref.name in no_arg:
            return True
        # Types defined in this file: check if they have params
        for td in self.ast.types:
            if td.name == ref.name:
                return len(td.params) == 0
        # Unknown types: assume they need params (safer)
        return False
