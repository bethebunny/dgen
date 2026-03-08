"""IR text deserialization.

Grammar classes compose via parser.read/try_read — no manual character dispatch.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

from dgen import Block, Constant, Dialect, Op, Type, TypeType, Value
from dgen.block import BlockArgument
from dgen.dialects import builtin
from dgen.module import ConstantOp, Module

_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_QUALIFIED = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*")
_STRING = re.compile(r'"[^"]*"')
_SSA = re.compile(r"%[a-zA-Z0-9_]+")
_NUMBER = re.compile(r"-?\d+\.\d*|-?\d*\.\d+|-?\d+")


def parse_module(text: str) -> Module:
    parser = ASMParser(text)
    while parser.try_read(ImportStatement) is not None:
        pass
    functions: list[builtin.FunctionOp] = []
    while not parser.done:
        op = parser.read(OpStatement)
        assert isinstance(op, builtin.FunctionOp)
        functions.append(op)
    return Module(functions=functions)


class Namespace:
    """Op/type registries populated from dialect imports."""

    def __init__(self) -> None:
        self.ops: dict[str, type[Op]] = {}
        self.types: dict[str, type[Type]] = {}
        b = Dialect.get("builtin")
        self.ops.update(b.ops)
        self.types.update(b.types)

    def import_dialect(self, name: str) -> None:
        for prefix in ("dgen.dialects", "toy.dialects"):
            try:
                importlib.import_module(f"{prefix}.{name}")
                break
            except ModuleNotFoundError:
                continue
        try:
            dialect = Dialect.get(name)
        except KeyError:
            raise RuntimeError(f"Unknown dialect: {name}") from None
        for k, v in dialect.ops.items():
            self.ops[f"{name}.{k}"] = v
        for k, v in dialect.types.items():
            self.types[f"{name}.{k}"] = v


class ASMParser:
    """Tokenizer with grammar-class dispatch via read/try_read."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos: int = 0
        self.namespace: Namespace = Namespace()
        self.name_table: dict[str, Value] = {}
        self.pending_ops: list[Op] = []

    @property
    def done(self) -> bool:
        self._skip_all()
        return self.pos >= len(self.text)

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def read(self, grammar: str | type) -> Any:  # noqa: ANN401
        if isinstance(grammar, str):
            self._skip_ws()
            self._expect(grammar)
            return grammar
        return grammar.read(self)

    def try_read(self, grammar: str | type) -> Any | None:  # noqa: ANN401
        saved = self.pos
        try:
            return self.read(grammar)
        except RuntimeError:
            self.pos = saved
            return None

    def parse_token(self, regex: re.Pattern[str]) -> str | None:
        self._skip_ws()
        if m := regex.match(self.text, self.pos):
            self.pos = m.end()
            return m.group()
        return None

    def expect_token(self, regex: re.Pattern[str], name: str) -> str:
        if (t := self.parse_token(regex)) is not None:
            return t
        raise RuntimeError(f"Expected {name} at {self.pos}")

    def raw_expr(self) -> object:
        """Parse a literal: int, float, string, [list], or {dict}."""
        if self.try_read("[") is not None:
            return self._finish_list(self.raw_expr)
        if self.try_read("{") is not None:
            return self._finish_dict()
        if (s := self.parse_token(_STRING)) is not None:
            return s[1:-1]
        if (n := self.parse_token(_NUMBER)) is not None:
            return float(n) if "." in n else int(n)
        raise RuntimeError(f"Expected literal at {self.pos}")

    def resolve(self, name: str) -> Value:
        if name not in self.name_table:
            self.name_table[name] = Value(name=name, type=builtin.Nil())
        return self.name_table[name]

    # -- Internal -----------------------------------------------------------

    def _finish_list(self, reader: object) -> list[object]:
        """Read comma-separated items until ']'. Opener already consumed."""
        items: list[object] = []
        if self.try_read("]") is None:
            items.append(reader())
            while self.try_read(",") is not None:
                items.append(reader())
            self.read("]")
        return items

    def _finish_dict(self) -> dict[str, object]:
        """Read key: value pairs until '}'. Opener already consumed."""
        d: dict[str, object] = {}
        if self.try_read("}") is None:
            k = self.expect_token(_STRING, "key")[1:-1]
            self.read(":")
            d[k] = self.raw_expr()
            while self.try_read(",") is not None:
                k = self.expect_token(_STRING, "key")[1:-1]
                self.read(":")
                d[k] = self.raw_expr()
            self.read("}")
        return d

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1

    def _skip_all(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def _skip_line(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] != "\n":
            self.pos += 1
        if self.pos < len(self.text):
            self.pos += 1

    def _expect(self, s: str) -> None:
        for ch in s:
            if self.pos >= len(self.text) or self.text[self.pos] != ch:
                raise RuntimeError(f"Expected '{s}' at {self.pos}")
            self.pos += 1

    def _parse_identifier(self) -> str:
        return self.expect_token(_IDENT, "identifier")

    def _parse_string_literal(self) -> str:
        return self.expect_token(_STRING, "string")[1:-1]

    def _skip_indent(self) -> int:
        indent = 0
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            indent += 1 if self.text[self.pos] == " " else 4
            self.pos += 1
        return indent


# -- Token grammar classes --------------------------------------------------


class SSAName:
    """Grammar: %identifier → str (without %)."""

    @classmethod
    def read(cls, parser: ASMParser) -> str:
        return parser.expect_token(_SSA, "SSA name")[1:]


class QualifiedName:
    """Grammar: ident or ident.ident → str."""

    @classmethod
    def read(cls, parser: ASMParser) -> str:
        return parser.expect_token(_QUALIFIED, "qualified name")


# -- Compound grammar classes -----------------------------------------------


class ImportStatement:
    @classmethod
    def read(cls, parser: ASMParser) -> str:
        parser._skip_all()
        word = parser._parse_identifier()
        if word == "from":
            parser._parse_identifier()  # "builtin"
            parser._parse_identifier()  # "import"
            parser._skip_line()
            return "builtin"
        if word == "import":
            name = parser._parse_identifier()
            parser._skip_line()
            parser.namespace.import_dialect(name)
            return name
        raise RuntimeError(f"Expected 'import' or 'from', got '{word}'")


class TypeExpression:
    """Grammar: %ssa | () | {dict} | Name[<params>]."""

    @classmethod
    def read(cls, parser: ASMParser) -> Type | Value:
        if (ssa := parser.try_read(SSAName)) is not None:
            return parser.resolve(ssa)
        if parser.try_read("()") is not None:
            return builtin.Nil()
        if parser.try_read("{") is not None:
            return parser._finish_dict()  # type: ignore[return-value]
        name = parser.read(QualifiedName)
        type_cls = parser.namespace.types.get(name)
        if type_cls is None:
            raise RuntimeError(f"Unknown type: {name}")
        if not type_cls.__params__:
            return type_cls()
        if parser.try_read("<") is None:
            return type_cls()  # all-default params
        kwargs = {}
        for i, (pname, ptype) in enumerate(type_cls.__params__):
            if i > 0:
                parser.read(",")
            kwargs[pname] = _param_value(parser, ptype)
        parser.read(">")
        return type_cls(**kwargs)


class OpExpression:
    """Grammar: name<params>(operands) [blocks] → (op_cls, kwargs)."""

    @classmethod
    def read(cls, parser: ASMParser) -> tuple[type[Op], dict[str, object]]:
        name = parser.read(QualifiedName)
        op_cls = parser.namespace.ops.get(name)
        if op_cls is None:
            raise RuntimeError(f"Unknown op: {name}")
        kwargs: dict[str, object] = {}
        if op_cls.__params__ and parser.try_read("<") is not None:
            for i, (pname, ptype) in enumerate(op_cls.__params__):
                if i > 0:
                    parser.read(",")
                kwargs[pname] = _param_value(parser, ptype)
            parser.read(">")
        parser.read("(")
        for i, (fname, ftype) in enumerate(op_cls.__operands__):
            if i > 0:
                parser.read(",")
            val = _operand(parser, ftype)
            if not isinstance(val, (Value, list)) and not issubclass(
                op_cls, ConstantOp
            ):
                val = _wrap(ftype, val)
            kwargs[fname] = val
        parser.read(")")
        for block_idx, block_name in enumerate(op_cls.__blocks__):
            if block_idx > 0:
                keyword = block_name.removesuffix("_body")
                parser._skip_all()
                saved = parser.pos
                try:
                    word = parser._parse_identifier()
                except RuntimeError:
                    parser.pos = saved
                    break
                if word != keyword:
                    parser.pos = saved
                    break
            args = parser.read(BlockArgs)
            parser.read(":")
            kwargs[block_name] = Block(ops=_indented_block(parser), args=args)
        return op_cls, kwargs


class OpStatement:
    """Grammar: %name [: Type] = (literal | op_expr). Returns Op."""

    @classmethod
    def read(cls, parser: ASMParser) -> Op:
        name = parser.read(SSAName)
        pre_type = (
            TypeExpression.read(parser) if parser.try_read(":") is not None else None
        )
        parser.read("=")
        # Try implicit constant (literal RHS)
        if (
            pre_type is not None
            and (value := parser.try_read(_ImplicitConstant)) is not None
        ):
            op = ConstantOp(name=name, value=value, type=pre_type)
            parser.name_table[name] = op
            return op
        op_cls, kwargs = parser.read(OpExpression)
        kwargs["name"] = name
        if pre_type is not None:
            kwargs["type"] = pre_type
        op = op_cls(**kwargs)
        parser.name_table[name] = op
        return op


class _ImplicitConstant:
    """Grammar: a raw literal in op-result position ({, [, -, or digit)."""

    @classmethod
    def read(cls, parser: ASMParser) -> object:
        parser._skip_ws()
        if parser.peek() in "{[-0123456789":
            return parser.raw_expr()
        raise RuntimeError("not a literal")


class BlockArgs:
    """Grammar: (%name: Type, ...) → list[BlockArgument]."""

    @classmethod
    def read(cls, parser: ASMParser) -> list[BlockArgument]:
        parser.read("(")
        args: list[BlockArgument] = []
        if parser.try_read(")") is not None:
            return args
        args.append(cls._one(parser))
        while parser.try_read(",") is not None:
            args.append(cls._one(parser))
        parser.read(")")
        return args

    @classmethod
    def _one(cls, parser: ASMParser) -> BlockArgument:
        name = parser.read(SSAName)
        parser.read(":")
        ty = TypeExpression.read(parser)
        arg = BlockArgument(name=name, type=ty)
        parser.name_table[name] = arg
        return arg


# -- Helpers ----------------------------------------------------------------


def _param_value(parser: ASMParser, field_type: type[Type]) -> Value | list[Value]:
    """Type-directed param: SSA ref, type (for TypeType), or literal → Constant."""
    if (ssa := parser.try_read(SSAName)) is not None:
        return parser.resolve(ssa)
    if field_type is Type or issubclass(field_type, TypeType):
        return TypeExpression.read(parser)
    raw = parser.raw_expr()
    if isinstance(raw, list) and not field_type.__params__:
        return [_wrap(field_type, v) if not isinstance(v, Value) else v for v in raw]
    return _wrap(field_type, raw)


def _wrap(field_type: type[Type], raw: object) -> Constant:
    """Wrap a raw Python value as a Constant of field_type."""
    if field_type.__params__:
        kwargs = {}
        for pname, ptype in field_type.__params__:
            assert isinstance(raw, list)
            kwargs[pname] = ptype().constant(len(raw))
        return field_type(**kwargs).constant(raw)
    return field_type().constant(raw)


def _operand(parser: ASMParser, field_type: type[Type]) -> object:
    """Parse an operand: SSA ref, () Nil, [list], or raw literal."""
    if (ssa := parser.try_read(SSAName)) is not None:
        return parser.resolve(ssa)
    if parser.try_read("()") is not None:
        return builtin.Nil()
    if parser.try_read("[") is not None:
        elems = parser._finish_list(lambda: _ssa_or_raw(parser))
        if any(isinstance(v, Value) for v in elems):
            return _pack(parser, elems, field_type)
        return elems
    return parser.raw_expr()


def _ssa_or_raw(parser: ASMParser) -> object:
    if (ssa := parser.try_read(SSAName)) is not None:
        return parser.resolve(ssa)
    return parser.raw_expr()


def _pack(
    parser: ASMParser, elems: list[object], field_type: type[Type]
) -> builtin.PackOp:
    """Expand a mixed list into a PackOp, wrapping raw literals as ConstantOps."""
    et = field_type()
    values: list[Value] = []
    for v in elems:
        if isinstance(v, Value):
            values.append(v)
        else:
            op = ConstantOp(value=v, type=et)
            parser.pending_ops.append(op)
            values.append(op)
    pack = builtin.PackOp(values=values, type=builtin.List(element_type=et))
    parser.pending_ops.append(pack)
    return pack


def _indented_block(parser: ASMParser) -> list[Op]:
    """After ':', skip to next line and parse ops at detected indent."""
    parser._skip_line()
    start = parser.pos
    indent = parser._skip_indent()
    parser.pos = start
    if indent == 0:
        return []
    return _block(parser, indent)


def _block(parser: ASMParser, min_indent: int) -> list[Op]:
    ops: list[Op] = []
    while parser.pos < len(parser.text):
        line_start = parser.pos
        indent = parser._skip_indent()
        if parser.pos >= len(parser.text) or parser.peek() == "\n":
            if parser.pos < len(parser.text):
                parser.pos += 1
            continue
        if indent < min_indent:
            parser.pos = line_start
            break
        eol = parser.text.find("\n", parser.pos)
        if eol == -1:
            eol = len(parser.text)
        op = OpStatement.read(parser)
        ops.extend(parser.pending_ops)
        parser.pending_ops.clear()
        ops.append(op)
        if parser.pos <= eol:
            parser._skip_line()
    return ops
