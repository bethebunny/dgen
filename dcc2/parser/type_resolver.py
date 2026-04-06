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

from collections.abc import Callable

from pycparser import c_ast

import dgen
from dgen.dialects.builtin import Array, Nil, String
from dgen.dialects.index import Index
from dgen.dialects.number import Float64
from dgen.module import pack

from dcc2.dialects import c_int, c_ptr, c_void
from dcc2.dialects.c import CFunctionType, Struct, StructField, Union
from dcc2.parser.c_literals import CONST_BINOPS, CONST_UNARY, parse_c_char, parse_c_int


class TypeResolverError(Exception):
    """An unresolvable C type."""


# ---------------------------------------------------------------------------
# Builtin C type tables (LP64 model)
# ---------------------------------------------------------------------------

# Maps C type-specifier strings to (bits, signed).
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
    "_Bool": (1, False),
    "bool": (1, False),
    "size_t": (64, False),
    "uintptr_t": (64, False),
    "ssize_t": (64, True),
    "ptrdiff_t": (64, True),
    "intptr_t": (64, True),
}

# Fixed-width integer typedefs (int8_t .. uint64_t).
for _prefix, _signed in [("int", True), ("uint", False)]:
    for _bits in (8, 16, 32, 64):
        _INT_WIDTHS[f"{_prefix}{_bits}_t"] = (_bits, _signed)

# Non-integer builtin types: name -> constructor.
_BUILTIN_TYPES: dict[str, Callable[[], dgen.Type]] = {
    "void": c_void,
    "float": Float64,
    "double": Float64,
    "long double": Float64,
}


# ---------------------------------------------------------------------------
# Constant expression evaluator
# ---------------------------------------------------------------------------

# Maps pycparser node type to evaluation function.
_CONST_EXPR_EVAL: dict[
    type[c_ast.Node],
    Callable[[c_ast.Node, dict[str, int]], int],
] = {}


def _const_eval_handler(
    node_type: type[c_ast.Node],
) -> Callable[
    [Callable[[c_ast.Node, dict[str, int]], int]],
    Callable[[c_ast.Node, dict[str, int]], int],
]:
    """Register a constant-expression evaluator for a pycparser node type."""

    def decorator(
        fn: Callable[[c_ast.Node, dict[str, int]], int],
    ) -> Callable[[c_ast.Node, dict[str, int]], int]:
        _CONST_EXPR_EVAL[node_type] = fn
        return fn

    return decorator


def eval_const_expr(node: c_ast.Node, enum_constants: dict[str, int]) -> int:
    """Evaluate a compile-time constant expression to int.

    Raises TypeResolverError on unsupported expressions.
    """
    handler = _CONST_EXPR_EVAL.get(type(node))
    if handler is not None:
        return handler(node, enum_constants)
    raise TypeResolverError(f"unsupported constant expression: {type(node).__name__}")


@_const_eval_handler(c_ast.Constant)
def _eval_constant(node: c_ast.Constant, _ec: dict[str, int]) -> int:
    if node.type == "int":
        return parse_c_int(node.value)
    if node.type == "char":
        return parse_c_char(node.value)
    return 0


@_const_eval_handler(c_ast.UnaryOp)
def _eval_unary(node: c_ast.UnaryOp, ec: dict[str, int]) -> int:
    fn = CONST_UNARY.get(node.op)
    if fn is not None:
        return fn(eval_const_expr(node.expr, ec))
    raise TypeResolverError(f"unsupported unary op in constant expression: {node.op}")


@_const_eval_handler(c_ast.BinaryOp)
def _eval_binary(node: c_ast.BinaryOp, ec: dict[str, int]) -> int:
    fn = CONST_BINOPS.get(node.op)
    if fn is not None:
        return fn(eval_const_expr(node.left, ec), eval_const_expr(node.right, ec))
    raise TypeResolverError(f"unsupported binary op in constant expression: {node.op}")


@_const_eval_handler(c_ast.Cast)
def _eval_cast(node: c_ast.Cast, ec: dict[str, int]) -> int:
    return eval_const_expr(node.expr, ec)


@_const_eval_handler(c_ast.ID)
def _eval_id(node: c_ast.ID, ec: dict[str, int]) -> int:
    if node.name in ec:
        return ec[node.name]
    raise TypeResolverError(f"unknown constant: {node.name}")


@_const_eval_handler(c_ast.TernaryOp)
def _eval_ternary(node: c_ast.TernaryOp, ec: dict[str, int]) -> int:
    cond = eval_const_expr(node.cond, ec)
    return eval_const_expr(node.iftrue if cond else node.iffalse, ec)


# ---------------------------------------------------------------------------
# TypeResolver
# ---------------------------------------------------------------------------


_RESOLVE_DISPATCH: dict[type[c_ast.Node], Callable[..., dgen.Type]] = {}


def _resolver(
    node_type: type[c_ast.Node],
) -> Callable[
    [Callable[..., dgen.Type]],
    Callable[..., dgen.Type],
]:
    """Register a TypeResolver method as the handler for a pycparser node type."""

    def decorator(fn: Callable[..., dgen.Type]) -> Callable[..., dgen.Type]:
        _RESOLVE_DISPATCH[node_type] = fn
        return fn

    return decorator


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
        handler = _RESOLVE_DISPATCH.get(type(node))
        if handler is not None:
            return handler(self, node)
        raise TypeResolverError(f"unsupported type node: {type(node).__name__}")

    # --- Dispatch targets ---

    @_resolver(c_ast.TypeDecl)
    def _resolve_type_decl(self, node: c_ast.TypeDecl) -> dgen.Type:
        return self.resolve(node.type)

    @_resolver(c_ast.Typename)
    def _resolve_typename(self, node: c_ast.Typename) -> dgen.Type:
        return self.resolve(node.type)

    @_resolver(c_ast.IdentifierType)
    def _resolve_identifier_type(self, node: c_ast.IdentifierType) -> dgen.Type:
        names = node.names
        joined = " ".join(names)

        if len(names) == 1 and names[0] in self.typedefs:
            return self.typedefs[names[0]]

        if joined in _INT_WIDTHS:
            bits, signed = _INT_WIDTHS[joined]
            return c_int(bits, signed)

        ctor = _BUILTIN_TYPES.get(joined)
        if ctor is not None:
            return ctor()

        raise TypeResolverError(f"unknown type: {joined}")

    @_resolver(c_ast.PtrDecl)
    def _resolve_ptr(self, node: c_ast.PtrDecl) -> dgen.Type:
        return c_ptr(self.resolve(node.type))

    @_resolver(c_ast.ArrayDecl)
    def _resolve_array(self, node: c_ast.ArrayDecl) -> dgen.Type:
        element = self.resolve(node.type)
        count = self._eval_array_dim(node.dim)
        return Array(element_type=element, n=Index().constant(count))

    @_resolver(c_ast.FuncDecl)
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

    @_resolver(c_ast.Struct)
    def _resolve_struct(self, node: c_ast.Struct) -> dgen.Type:
        """Resolve a struct definition or forward reference."""
        tag = node.name or self._unique_anon_tag()

        # Forward reference to previously-defined struct.
        if tag in self.structs and node.decls is None:
            return self.structs[tag]

        fields = self._resolve_fields(node.decls or [])
        struct_type = Struct(
            tag=String().constant(tag),
            fields=pack(fields),
        )
        self.structs[tag] = struct_type
        return struct_type

    @_resolver(c_ast.Union)
    def _resolve_union(self, node: c_ast.Union) -> dgen.Type:
        """Resolve a union definition or forward reference."""
        tag = node.name or self._unique_anon_tag()

        if tag in self.unions and node.decls is None:
            return self.unions[tag]

        fields = self._resolve_fields(node.decls or [])
        union_type = Union(
            tag=String().constant(tag),
            fields=pack(fields),
        )
        self.unions[tag] = union_type
        return union_type

    @_resolver(c_ast.Enum)
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

    # --- Helpers ---

    def _adjust_param_type(self, ptype: dgen.Type) -> dgen.Type:
        """C11 6.7.6.3p7-8: array params -> pointer, function params -> pointer-to-function."""
        if isinstance(ptype, Array):
            return c_ptr(ptype.element_type)
        if isinstance(ptype, CFunctionType):
            return c_ptr(ptype)
        return ptype

    def _unique_anon_tag(self) -> str:
        """Generate a unique tag for anonymous struct/union."""
        self._anon_counter += 1
        return f"_anon_{self._anon_counter}"

    def _resolve_fields(self, decls: list[c_ast.Node]) -> list[StructField]:
        """Resolve struct/union field declarations to StructField instances."""
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

    def register_typedef(self, name: str, target_type: dgen.Type) -> None:
        self.typedefs[name] = target_type

    def _eval_array_dim(self, dim: c_ast.Node | None) -> int:
        if dim is None:
            return 0
        return eval_const_expr(dim, self.enum_constants)
