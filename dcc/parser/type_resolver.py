"""Resolve pycparser type AST nodes to dgen C dialect types."""

from __future__ import annotations

from pycparser import c_ast

from dcc.parser.c_literals import _CONST_BINOPS, _CONST_UNARY, parse_c_char, parse_c_int

import dgen
from dgen.dialects.index import Index
from dgen.dialects.builtin import String
from dgen.dialects.builtin import Array
from dgen.dialects.function import Function
from dgen.module import pack
from dgen.dialects.number import Float64
from dcc.dialects import c_int, c_ptr, c_void
from dcc.dialects.c import CStruct, CUnion


# Standard C integer type widths (LP64 model)
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

_FLOAT_KINDS: dict[str, int] = {
    "float": 0,
    "double": 1,
    "long double": 2,
}


class TypeResolver:
    """Resolve pycparser type AST nodes to dgen C dialect types.

    Maintains a registry of typedefs and struct/union definitions.
    """

    def __init__(self) -> None:
        self.typedefs: dict[str, dgen.Type] = {}
        self.structs: dict[str, CStruct] = {}
        self.unions: dict[str, CUnion] = {}
        self.enums: dict[str, dgen.Type] = {}
        self.enum_constants: dict[str, int] = {}
        # struct tag -> list of (field_name, field_type) for member access
        self.struct_fields: dict[str, list[tuple[str, dgen.Type]]] = {}

    def resolve(self, node: c_ast.Node) -> dgen.Type:
        """Resolve a pycparser type node to a dgen C dialect type."""
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

        # Unknown — treat as opaque i64
        return c_int(64, signed=True)

    def _resolve_func_decl(self, node: c_ast.FuncDecl) -> dgen.Type:
        """Resolve a FuncDecl to a Function type."""
        ret_type = self.resolve(node.type)
        # TODO: derive argument types from the FuncDecl parameters
        return Function(arguments=pack(), result_type=ret_type)

    def _resolve_struct(self, node: c_ast.Struct) -> dgen.Type:
        """Resolve a struct definition or forward reference."""
        tag = node.name or "_anon"

        if tag in self.structs and node.decls is None:
            return self.structs[tag]

        field_list: list[tuple[str, dgen.Type]] = []
        if node.decls is not None:
            for decl in node.decls:
                if isinstance(decl, c_ast.Decl):
                    fname = decl.name or "_pad"
                    ftype = self.resolve(decl.type)
                    field_list.append((fname, ftype))

        struct_type = CStruct(
            tag_name=String().constant(tag),
            field_names=String().constant(tag),
            field_types=String().constant(tag),
        )
        self.structs[tag] = struct_type
        self.struct_fields[tag] = field_list
        return struct_type

    def _resolve_union(self, node: c_ast.Union) -> dgen.Type:
        """Resolve a union definition or forward reference."""
        tag = node.name or "_anon"

        if tag in self.unions and node.decls is None:
            return self.unions[tag]

        union_type = CUnion(
            tag_name=String().constant(tag),
            field_names=String().constant(tag),
            field_types=String().constant(tag),
        )
        self.unions[tag] = union_type
        return union_type

    def _resolve_enum(self, node: c_ast.Enum) -> dgen.Type:
        """Resolve an enum — enums are lowered to int."""
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

    def get_struct_field_type(
        self, struct_type: dgen.Type, field_name: str
    ) -> dgen.Type:
        """Look up a struct field's type by name."""
        if isinstance(struct_type, CStruct):
            tag = struct_type.tag_name.__constant__.to_json()
            assert isinstance(tag, str)
            fields = self.struct_fields.get(tag, [])
            for fname, ftype in fields:
                if fname == field_name:
                    return ftype
        # Unknown field — return an opaque pointer so chained -> accesses
        # and subscripts can continue. sqlite3 forward-declares structs
        # and uses nested field access heavily.
        return c_ptr(c_void())

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
            fn = _CONST_UNARY.get(node.op)
            return fn(self._eval_const_expr(node.expr)) if fn else 0
        if isinstance(node, c_ast.BinaryOp):
            fn = _CONST_BINOPS.get(node.op)
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
