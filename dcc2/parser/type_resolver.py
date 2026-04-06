"""Resolve pycparser type AST nodes to dgen types.

Responsibility:
- Map C type syntax to dgen type values
- Track typedef/struct/union/enum definitions
- Resolve forward references for struct/union
- Apply C11 6.7.6.3 parameter type adjustments

Not responsible for:
- Type checking (compatibility, assignability)
- Struct layout computation (offsets are 0; filled by layout pass)
- Type qualifiers (const, volatile, restrict)
"""

from __future__ import annotations

from pycparser import c_ast

import dgen
from dgen.dialects.builtin import Array, Nil, String
from dgen.dialects.index import Index
from dgen.dialects.number import Float64
from dgen.module import pack

from dcc2.dialects import c_int, c_ptr, c_void
from dcc2.dialects.c import CFunctionType, Struct, StructField, Union
from dcc2.parser.c_literals import CONST_BINOPS, CONST_UNARY, parse_c_char, parse_c_int

# Standard C integer type widths (LP64 model).
_INT_WIDTHS: dict[str, tuple[int, bool]] = {
    "char": (8, True),
    "signed char": (8, True),
    "unsigned char": (8, False),
    "short": (16, True),
    "signed short": (16, True),
    "unsigned short": (16, False),
    "int": (32, True),
    "signed int": (32, True),
    "unsigned int": (32, False),
    "long": (64, True),
    "signed long": (64, True),
    "unsigned long": (64, False),
    "long long": (64, True),
    "signed long long": (64, True),
    "unsigned long long": (64, False),
    "signed": (32, True),
    "unsigned": (32, False),
}

# Fixed-width integer typedefs.
_FIXED_WIDTH: dict[str, tuple[int, bool]] = {}
for _prefix, _signed in [("int", True), ("uint", False)]:
    for _bits in (8, 16, 32, 64):
        _FIXED_WIDTH[f"{_prefix}{_bits}_t"] = (_bits, _signed)

_FLOAT_KINDS: set[str] = {"float", "double", "long double"}

_PLATFORM_TYPES: dict[str, tuple[int, bool]] = {
    "size_t": (64, False),
    "uintptr_t": (64, False),
    "ssize_t": (64, True),
    "ptrdiff_t": (64, True),
    "intptr_t": (64, True),
}


class TypeResolverError(Exception):
    """An unresolvable C type."""


class TypeResolver:
    """Resolve pycparser type AST nodes to dgen types.

    Maintains a registry of typedefs, struct/union/enum definitions.
    Struct field metadata is stored in the Struct type itself via StructField
    instances, not in a side-channel dict.
    """

    def __init__(self) -> None:
        self.typedefs: dict[str, dgen.Type] = {}
        self.structs: dict[str, Struct] = {}
        self.unions: dict[str, Union] = {}
        self.enum_constants: dict[str, int] = {}
        self._anon_counter: int = 0

    def resolve(self, node: c_ast.Node) -> dgen.Type:
        """Resolve a pycparser type node to a dgen type."""
        if isinstance(node, c_ast.TypeDecl):
            return self.resolve(node.type)

        if isinstance(node, c_ast.IdentifierType):
            return self._resolve_identifier_type(node.names)

        if isinstance(node, c_ast.PtrDecl):
            pointee = self.resolve(node.type)
            return c_ptr(pointee)

        if isinstance(node, c_ast.ArrayDecl):
            element = self.resolve(node.type)
            count = self._eval_array_dim(node.dim)
            return Array(element_type=element, n=Index().constant(count))

        if isinstance(node, c_ast.FuncDecl):
            return self._resolve_func_decl(node)

        if isinstance(node, c_ast.Struct):
            return self._resolve_struct(node)

        if isinstance(node, c_ast.Union):
            return self._resolve_union(node)

        if isinstance(node, c_ast.Enum):
            return self._resolve_enum(node)

        if isinstance(node, c_ast.Typename):
            return self.resolve(node.type)

        return c_void()

    def _resolve_identifier_type(self, names: list[str]) -> dgen.Type:
        """Resolve an IdentifierType (e.g. ['unsigned', 'int'])."""
        joined = " ".join(names)

        if len(names) == 1 and names[0] in self.typedefs:
            return self.typedefs[names[0]]

        if joined in _INT_WIDTHS:
            bits, signed = _INT_WIDTHS[joined]
            return c_int(bits, signed)

        if joined in _FLOAT_KINDS:
            return Float64()

        if joined == "void":
            return c_void()

        if joined in ("_Bool", "bool"):
            return c_int(1, signed=False)

        if joined in _PLATFORM_TYPES:
            bits, signed = _PLATFORM_TYPES[joined]
            return c_int(bits, signed)

        if joined in _FIXED_WIDTH:
            bits, signed = _FIXED_WIDTH[joined]
            return c_int(bits, signed)

        raise TypeResolverError(f"unknown type: {joined}")

    def _adjust_param_type(self, ptype: dgen.Type) -> dgen.Type:
        """C11 6.7.6.3p7-8: array params -> pointer, function params -> pointer-to-function."""
        if isinstance(ptype, Array):
            return c_ptr(ptype.element_type)
        if isinstance(ptype, CFunctionType):
            return c_ptr(ptype)
        return ptype

    def _resolve_func_decl(self, node: c_ast.FuncDecl) -> dgen.Type:
        """Resolve a FuncDecl to a CFunctionType."""
        ret_type = self.resolve(node.type)

        arg_types: list[dgen.Type] = []
        is_variadic = False

        if node.args is not None:
            for param in node.args.params or []:
                if isinstance(param, c_ast.EllipsisParam):
                    is_variadic = True
                    continue
                if isinstance(param, (c_ast.Decl, c_ast.Typename)):
                    ptype = self.resolve(
                        param.type if isinstance(param, c_ast.Decl) else param
                    )
                    arg_types.append(self._adjust_param_type(ptype))

            # C11: f(void) means no parameters.
            if len(arg_types) == 1 and isinstance(arg_types[0], Nil):
                arg_types = []

        n_fixed = len(arg_types)
        idx = Index()
        return CFunctionType(
            arguments=pack(arg_types),
            result_type=ret_type,
            is_variadic=idx.constant(int(is_variadic)),
            n_fixed_params=idx.constant(n_fixed),
        )

    def _unique_anon_tag(self) -> str:
        """Generate a unique tag for anonymous struct/union."""
        self._anon_counter += 1
        return f"_anon_{self._anon_counter}"

    def _resolve_fields(self, decls: list[c_ast.Node] | None) -> list[StructField]:
        """Resolve struct/union field declarations to StructField instances."""
        if decls is None:
            return []
        fields: list[StructField] = []
        idx = Index()
        for decl in decls:
            if isinstance(decl, c_ast.Decl):
                fname = decl.name or "_pad"
                ftype = self.resolve(decl.type)
                fields.append(
                    StructField(
                        field_name=String().constant(fname),
                        field_type=ftype,
                        offset=idx.constant(0),  # filled by layout pass
                    )
                )
        return fields

    def _resolve_struct(self, node: c_ast.Struct) -> dgen.Type:
        """Resolve a struct definition or forward reference."""
        tag = node.name or self._unique_anon_tag()

        # Forward reference to previously-defined struct.
        if tag in self.structs and node.decls is None:
            return self.structs[tag]

        fields = self._resolve_fields(node.decls)
        struct_type = Struct(
            tag=String().constant(tag),
            fields=pack(fields),
        )
        self.structs[tag] = struct_type
        return struct_type

    def _resolve_union(self, node: c_ast.Union) -> dgen.Type:
        """Resolve a union definition or forward reference."""
        tag = node.name or self._unique_anon_tag()

        if tag in self.unions and node.decls is None:
            return self.unions[tag]

        fields = self._resolve_fields(node.decls)
        union_type = Union(
            tag=String().constant(tag),
            fields=pack(fields),
        )
        self.unions[tag] = union_type
        return union_type

    def _resolve_enum(self, node: c_ast.Enum) -> dgen.Type:
        """Resolve an enum -- enums are lowered to int."""
        if node.values is not None:
            val = 0
            for enumerator in node.values.enumerators:
                if enumerator.value is not None:
                    val = eval_const_expr(enumerator.value, self.enum_constants)
                self.enum_constants[enumerator.name] = val
                val += 1

        return c_int(32, signed=True)

    def register_typedef(self, name: str, target_type: dgen.Type) -> None:
        self.typedefs[name] = target_type

    def _eval_array_dim(self, dim: c_ast.Node | None) -> int:
        if dim is None:
            return 0
        return eval_const_expr(dim, self.enum_constants)


def eval_const_expr(node: c_ast.Node, enum_constants: dict[str, int]) -> int:
    """Evaluate a compile-time constant expression to int.

    Extracted from TypeResolver so it can be reused and tested independently.
    Raises TypeResolverError on unsupported expressions.
    """
    if isinstance(node, c_ast.Constant):
        if node.type == "int":
            return parse_c_int(node.value)
        if node.type == "char":
            return parse_c_char(node.value)
        return 0
    if isinstance(node, c_ast.UnaryOp):
        fn = CONST_UNARY.get(node.op)
        if fn is not None:
            return fn(eval_const_expr(node.expr, enum_constants))
        raise TypeResolverError(
            f"unsupported unary op in constant expression: {node.op}"
        )
    if isinstance(node, c_ast.BinaryOp):
        fn = CONST_BINOPS.get(node.op)
        if fn is not None:
            return fn(
                eval_const_expr(node.left, enum_constants),
                eval_const_expr(node.right, enum_constants),
            )
        raise TypeResolverError(
            f"unsupported binary op in constant expression: {node.op}"
        )
    if isinstance(node, c_ast.Cast):
        return eval_const_expr(node.expr, enum_constants)
    if isinstance(node, c_ast.ID):
        if node.name in enum_constants:
            return enum_constants[node.name]
        raise TypeResolverError(f"unknown constant: {node.name}")
    if isinstance(node, c_ast.TernaryOp):
        cond = eval_const_expr(node.cond, enum_constants)
        return eval_const_expr(node.iftrue if cond else node.iffalse, enum_constants)
    raise TypeResolverError(f"unsupported constant expression: {type(node).__name__}")
