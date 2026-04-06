"""Resolve pycparser type AST nodes to dgen types."""

from __future__ import annotations

from pycparser import c_ast

import dgen
from dgen.dialects.builtin import Array, String
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

_FLOAT_KINDS: set[str] = {"float", "double", "long double"}


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
        self.enums: dict[str, dgen.Type] = {}
        self.enum_constants: dict[str, int] = {}

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

        if joined in ("size_t", "uintptr_t"):
            return c_int(64, signed=False)
        if joined in ("ssize_t", "ptrdiff_t", "intptr_t"):
            return c_int(64, signed=True)

        for prefix, s in [("int", True), ("uint", False)]:
            for bits in (8, 16, 32, 64):
                if joined == f"{prefix}{bits}_t":
                    return c_int(bits, s)

        # Unknown -- treat as opaque i64.
        return c_int(64, signed=True)

    def _resolve_func_decl(self, node: c_ast.FuncDecl) -> dgen.Type:
        """Resolve a FuncDecl to a CFunctionType."""
        ret_type = self.resolve(node.type)

        arg_types: list[dgen.Type] = []
        is_variadic = False
        n_fixed = 0

        if node.args is not None:
            for param in node.args.params or []:
                if isinstance(param, c_ast.EllipsisParam):
                    is_variadic = True
                    continue
                if isinstance(param, c_ast.Decl):
                    ptype = self.resolve(param.type)
                    # C11 6.7.6.3p7-8: array params adjusted to pointer,
                    # function params adjusted to pointer-to-function.
                    if isinstance(ptype, Array):
                        ptype = c_ptr(ptype.element_type)
                    elif isinstance(ptype, CFunctionType):
                        ptype = c_ptr(ptype)
                    arg_types.append(ptype)
                elif isinstance(param, c_ast.Typename):
                    ptype = self.resolve(param)
                    if isinstance(ptype, Array):
                        ptype = c_ptr(ptype.element_type)
                    elif isinstance(ptype, CFunctionType):
                        ptype = c_ptr(ptype)
                    arg_types.append(ptype)

            # Check for f(void) -- single void param means no args.
            if (
                len(arg_types) == 1
                and isinstance(arg_types[0], type(c_void()))
                and arg_types[0].__class__.__name__ == "Nil"
            ):
                arg_types = []

        n_fixed = len(arg_types)
        idx = Index()
        return CFunctionType(
            arguments=pack(arg_types),
            result_type=ret_type,
            is_variadic=idx.constant(int(is_variadic)),
            n_fixed_params=idx.constant(n_fixed),
        )

    def _resolve_struct(self, node: c_ast.Struct) -> dgen.Type:
        """Resolve a struct definition or forward reference."""
        tag = node.name or "_anon"

        if tag in self.structs and node.decls is None:
            return self.structs[tag]

        fields: list[StructField] = []
        if node.decls is not None:
            idx = Index()
            for decl in node.decls:
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

        struct_type = Struct(
            tag=String().constant(tag),
            fields=pack(fields),
        )
        self.structs[tag] = struct_type
        return struct_type

    def _resolve_union(self, node: c_ast.Union) -> dgen.Type:
        """Resolve a union definition or forward reference."""
        tag = node.name or "_anon"

        if tag in self.unions and node.decls is None:
            return self.unions[tag]

        fields: list[StructField] = []
        if node.decls is not None:
            idx = Index()
            for decl in node.decls:
                if isinstance(decl, c_ast.Decl):
                    fname = decl.name or "_pad"
                    ftype = self.resolve(decl.type)
                    fields.append(
                        StructField(
                            field_name=String().constant(fname),
                            field_type=ftype,
                            offset=idx.constant(0),  # all fields at offset 0
                        )
                    )

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
                    val = self._eval_const_expr(enumerator.value)
                self.enum_constants[enumerator.name] = val
                val += 1

        return c_int(32, signed=True)

    def register_typedef(self, name: str, target_type: dgen.Type) -> None:
        self.typedefs[name] = target_type

    def _eval_array_dim(self, dim: c_ast.Node | None) -> int:
        if dim is None:
            return 0
        return self._eval_const_expr(dim)

    def _eval_const_expr(self, node: c_ast.Node) -> int:
        """Evaluate a compile-time constant expression to int."""
        if isinstance(node, c_ast.Constant):
            if node.type == "int":
                return parse_c_int(node.value)
            if node.type == "char":
                return parse_c_char(node.value)
            return 0
        if isinstance(node, c_ast.UnaryOp):
            fn = CONST_UNARY.get(node.op)
            return fn(self._eval_const_expr(node.expr)) if fn else 0
        if isinstance(node, c_ast.BinaryOp):
            fn = CONST_BINOPS.get(node.op)
            if fn is not None:
                return fn(
                    self._eval_const_expr(node.left), self._eval_const_expr(node.right)
                )
            return 0
        if isinstance(node, c_ast.Cast):
            return self._eval_const_expr(node.expr)
        if isinstance(node, c_ast.ID):
            return self.enum_constants.get(node.name, 0)
        if isinstance(node, c_ast.TernaryOp):
            cond = self._eval_const_expr(node.cond)
            return self._eval_const_expr(node.iftrue if cond else node.iffalse)
        return 0
