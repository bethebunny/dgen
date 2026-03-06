"""Parser for .dgen dialect specification files."""

from __future__ import annotations

from dgen.gen.ast import (
    Constraint,
    DataField,
    DgenFile,
    ImportDecl,
    OpDecl,
    OperandDecl,
    ParamDecl,
    StaticField,
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
            elif line.startswith("import "):
                parts = line.split()
                if len(parts) != 2:
                    raise SyntaxError(f"expected 'import module', got: {line!r}")
                result.imports.append(ImportDecl(module=parts[1], names=[]))
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
        rest = line[6:]  # strip "trait "
        has_body = rest.rstrip().endswith(":")
        if has_body:
            name = rest.rstrip()[:-1].strip()
            statics = self._parse_trait_body()
        else:
            name = rest.strip()
            statics = []
        return TraitDecl(name=name, statics=statics)

    def _parse_trait_body(self) -> list[StaticField]:
        """Parse indented trait body lines, return static fields."""
        statics: list[StaticField] = []
        while self.pos + 1 < len(self.lines):
            next_line = self.lines[self.pos + 1]
            if not next_line or not next_line[0].isspace():
                break
            self.pos += 1
            stripped = next_line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if stripped.startswith("static "):
                statics.append(_parse_static_field(stripped))
            else:
                raise SyntaxError(f"unexpected line in trait body: {stripped!r}")
        return statics

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

        data: list[DataField] = []
        layout: str | None = None
        traits: list[str] = []
        statics: list[StaticField] = []
        if has_body:
            data, layout, traits, statics = self._parse_type_body()
        return TypeDecl(
            name=name,
            params=params,
            data=data,
            layout=layout,
            traits=traits,
            statics=statics,
        )

    def _parse_type_body(
        self,
    ) -> tuple[list[DataField], str | None, list[str], list[StaticField]]:
        """Parse indented type body lines, return (data fields, layout name, traits, statics)."""
        data: list[DataField] = []
        layout = None
        traits: list[str] = []
        statics: list[StaticField] = []
        while self.pos + 1 < len(self.lines):
            next_line = self.lines[self.pos + 1]
            if not next_line or not next_line[0].isspace():
                break
            self.pos += 1
            stripped = next_line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            # Layout declaration: layout LayoutName
            if stripped.startswith("layout "):
                layout = stripped.split()[1]
                continue
            # Trait declaration: has trait Name
            if stripped.startswith("has trait "):
                traits.append(stripped.split()[2])
                continue
            # Static field: static name: Type [= default]
            if stripped.startswith("static "):
                statics.append(_parse_static_field(stripped))
                continue
            # Field declaration: name: TypeExpr
            if ":" in stripped:
                colon = stripped.index(":")
                field_name = stripped[:colon].strip()
                type_str = stripped[colon + 1 :].strip()
                data.append(DataField(name=field_name, type=_parse_type_ref(type_str)))
        return data, layout, traits, statics

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
        return_type: TypeRef | None = None
        if "->" in rest:
            ret_str = rest[rest.index("->") + 2 :].strip()
            return_type = _parse_type_ref(ret_str)

        blocks: list[str] = []
        traits: list[str] = []
        constraints: list[Constraint] = []
        if has_body:
            blocks, traits, constraints = self._parse_op_body()
        return OpDecl(
            name=name,
            params=params,
            operands=operands,
            return_type=return_type,
            blocks=blocks,
            traits=traits,
            constraints=constraints,
        )

    def _parse_op_body(self) -> tuple[list[str], list[str], list[Constraint]]:
        """Parse indented op body lines, return (block names, traits, constraints)."""
        blocks: list[str] = []
        traits: list[str] = []
        constraints: list[Constraint] = []
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
            elif stripped.startswith("has trait "):
                traits.append(stripped.split()[2])
            elif stripped.startswith("requires "):
                constraints.append(_parse_constraint(stripped))
        return blocks, traits, constraints


def _parse_static_field(line: str) -> StaticField:
    """Parse a 'static name: Type [= default]' line into a StaticField."""
    rest = line[7:]  # strip "static "
    default: str | None = None
    if "=" in rest:
        rest, default_str = rest.rsplit("=", 1)
        default = default_str.strip()
        rest = rest.strip()
    if ":" not in rest:
        raise SyntaxError(f"static field requires type annotation: {line!r}")
    name, type_str = rest.split(":", 1)
    return StaticField(
        name=name.strip(),
        type=_parse_type_ref(type_str.strip()),
        default=default,
    )


def _parse_constraint(line: str) -> Constraint:
    """Parse a 'requires ...' line into a Constraint."""
    rest = line[9:]  # strip "requires "
    if " ~= " in rest:
        lhs, pattern = rest.split(" ~= ", 1)
        return Constraint(kind="match", lhs=lhs.strip(), pattern=pattern.strip())
    if " == " in rest:
        lhs, rhs = rest.split(" == ", 1)
        lhs_s, rhs_s = lhs.strip(), rhs.strip()
        # Only "eq" if both sides are simple metavariables ($Var, no dots)
        if (
            lhs_s.startswith("$")
            and "." not in lhs_s
            and rhs_s.startswith("$")
            and "." not in rhs_s
        ):
            return Constraint(kind="eq", lhs=lhs_s, rhs=rhs_s)
    return Constraint(kind="expr", expr=rest.strip())


def _parse_decl_parts(
    s: str,
) -> list[tuple[str, TypeRef | None, str | None, bool]]:
    """Parse comma-separated declarations into (name, type_ref, default, variadic) tuples.

    Each declaration has the form: name[: Type] [= default].
    The type annotation is optional; when absent, type_ref is None.
    """
    parts: list[tuple[str, TypeRef | None, str | None, bool]] = []
    for part in _split_commas(s):
        part = part.strip()
        default: str | None = None
        if "=" in part:
            decl, default = part.rsplit("=", 1)
            default = default.strip()
            part = decl.strip()
        type_ref: TypeRef | None = None
        variadic = False
        if ":" in part:
            name, type_str = part.split(":", 1)
            name = name.strip()
            type_ref = _parse_type_ref(type_str.strip())
            if type_ref.name == "list":
                variadic = True
                if type_ref.args:
                    type_ref = type_ref.args[0]
                else:
                    type_ref = None
        else:
            name = part.strip()
        parts.append((name, type_ref, default, variadic))
    return parts


def _parse_params(s: str) -> list[ParamDecl]:
    """Parse a comma-separated parameter list: name: Type [= default], ..."""
    params: list[ParamDecl] = []
    for name, type_ref, default, variadic in _parse_decl_parts(s):
        if type_ref is None:
            raise SyntaxError(f"parameter {name!r} requires a type annotation")
        params.append(
            ParamDecl(name=name, type=type_ref, default=default, variadic=variadic)
        )
    return params


def _parse_operands(s: str) -> list[OperandDecl]:
    """Parse a comma-separated operand list: name[: Type] [= default], ..."""
    return [
        OperandDecl(name=name, type=type_ref, default=default, variadic=variadic)
        for name, type_ref, default, variadic in _parse_decl_parts(s)
    ]


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
