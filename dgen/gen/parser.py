"""Parser for .dgen dialect specification files."""

from __future__ import annotations

from dgen.gen.ast import (
    DgenFile,
    ImportDecl,
    LayoutExpr,
    OpDecl,
    OperandDecl,
    ParamDecl,
    TraitDecl,
    TypeDecl,
    TypeRef,
)


def parse(source: str) -> DgenFile:
    """Parse a .dgen source string into a DgenFile AST."""
    return _Parser(source).parse()


class _Parser:
    def __init__(self, source: str) -> None:
        self.lines = source.splitlines()
        self.pos = 0

    def parse(self) -> DgenFile:
        result = DgenFile()
        while self.pos < len(self.lines):
            line = self.lines[self.pos].strip()
            if not line or line.startswith("#"):
                self.pos += 1
                continue
            if line.startswith("from "):
                result.imports.append(self._parse_import(line))
            elif line.startswith("trait "):
                result.traits.append(self._parse_trait(line))
            elif line.startswith("type "):
                result.types.append(self._parse_type(line))
            elif line.startswith("op "):
                result.ops.append(self._parse_op(line))
            else:
                raise SyntaxError(f"unexpected line: {line!r}")
            self.pos += 1
        return result

    def _parse_import(self, line: str) -> ImportDecl:
        # from module import Name1, Name2
        parts = line.split()
        module = parts[1]
        names = [n.strip(",") for n in parts[3:]]
        return ImportDecl(module=module, names=names)

    def _parse_trait(self, line: str) -> TraitDecl:
        return TraitDecl(name=line.split()[1])

    def _parse_type(self, line: str) -> TypeDecl:
        rest = line[5:]  # strip "type "
        params: list[ParamDecl] = []

        # Check for colon at end (body follows)
        has_body = rest.rstrip().endswith(":")
        if has_body:
            rest = rest.rstrip()[:-1]  # strip trailing ':'

        # Parse name and optional params
        if "<" in rest:
            lt = rest.index("<")
            name = rest[:lt].strip()
            gt = _find_matching(rest, lt, "<", ">")
            param_str = rest[lt + 1 : gt]
            params = _parse_params(param_str)
        else:
            name = rest.strip()

        layout = self._parse_type_body() if has_body else None
        return TypeDecl(name=name, params=params, layout=layout)

    def _parse_type_body(self) -> LayoutExpr | None:
        """Parse indented type body lines, return layout if found."""
        layout = None
        while self.pos + 1 < len(self.lines):
            next_line = self.lines[self.pos + 1]
            if not next_line or not next_line[0].isspace():
                break
            self.pos += 1
            stripped = next_line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if stripped.startswith("layout "):
                layout = _parse_layout(stripped[7:])
        return layout

    def _parse_op(self, line: str) -> OpDecl:
        rest = line[3:]  # strip "op "
        params: list[ParamDecl] = []
        operands: list[OperandDecl] = []

        # Check for body
        has_body = rest.rstrip().endswith(":")
        if has_body:
            rest = rest.rstrip()[:-1]

        # Parse name (up to first < or ()
        name_end = len(rest)
        for ch in "<(":
            idx = rest.find(ch)
            if idx != -1 and idx < name_end:
                name_end = idx
        name = rest[:name_end].strip()
        rest = rest[name_end:]

        # Parse optional params <...>
        if rest.startswith("<"):
            gt = _find_matching(rest, 0, "<", ">")
            param_str = rest[1:gt]
            params = _parse_params(param_str)
            rest = rest[gt + 1 :]

        # Parse operands (...)
        if rest.startswith("("):
            cp = rest.index(")")
            operand_str = rest[1:cp].strip()
            if operand_str:
                operands = _parse_operands(operand_str)
            rest = rest[cp + 1 :]

        # Parse return type -> Type
        return_type = TypeRef("Type")
        if "->" in rest:
            ret_str = rest[rest.index("->") + 2 :].strip()
            return_type = _parse_type_ref(ret_str)

        blocks = self._parse_op_body() if has_body else []
        return OpDecl(
            name=name,
            params=params,
            operands=operands,
            return_type=return_type,
            blocks=blocks,
        )

    def _parse_op_body(self) -> list[str]:
        """Parse indented op body lines, return block names."""
        blocks: list[str] = []
        while self.pos + 1 < len(self.lines):
            next_line = self.lines[self.pos + 1]
            if not next_line or not next_line[0].isspace():
                break
            self.pos += 1
            stripped = next_line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if stripped.startswith("block "):
                blocks.append(stripped.split()[1])
        return blocks


def _parse_params(s: str) -> list[ParamDecl]:
    """Parse a comma-separated parameter list: name: Type [= default], ..."""
    params: list[ParamDecl] = []
    for part in _split_commas(s):
        part = part.strip()
        default = None
        if "=" in part:
            decl, default = part.rsplit("=", 1)
            default = default.strip()
            part = decl.strip()
        name, type_str = part.split(":", 1)
        params.append(
            ParamDecl(
                name=name.strip(),
                type=_parse_type_ref(type_str.strip()),
                default=default,
            )
        )
    return params


def _parse_operands(s: str) -> list[OperandDecl]:
    """Parse a comma-separated operand list: name: Type [= default], ..."""
    operands: list[OperandDecl] = []
    for part in _split_commas(s):
        part = part.strip()
        default = None
        if "=" in part:
            decl, default = part.rsplit("=", 1)
            default = default.strip()
            part = decl.strip()
        name, type_str = part.split(":", 1)
        operands.append(
            OperandDecl(
                name=name.strip(),
                type=_parse_type_ref(type_str.strip()),
                default=default,
            )
        )
    return operands


def _parse_type_ref(s: str) -> TypeRef:
    """Parse a type reference: Name, Name<args>, list<T>, or Type."""
    s = s.strip()
    if "<" in s:
        lt = s.index("<")
        name = s[:lt]
        gt = _find_matching(s, lt, "<", ">")
        args_str = s[lt + 1 : gt]
        args = [_parse_type_ref(a.strip()) for a in _split_commas(args_str)]
        return TypeRef(name=name, args=args)
    return TypeRef(name=s)


def _parse_layout(s: str) -> LayoutExpr:
    """Parse a layout expression: INT, Array<INT, rank>, FatPointer<BYTE>."""
    s = s.strip()
    if "<" in s:
        lt = s.index("<")
        name = s[:lt]
        args_str = s[lt + 1 : s.rindex(">")]
        args = [a.strip() for a in args_str.split(",")]
        return LayoutExpr(name=name, args=args)
    return LayoutExpr(name=s)


def _split_commas(s: str) -> list[str]:
    """Split on commas respecting <> nesting."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _find_matching(s: str, start: int, open_ch: str, close_ch: str) -> int:
    """Find the matching close bracket starting from position start."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise SyntaxError(f"unmatched {open_ch} in {s!r}")
