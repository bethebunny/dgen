"""Parse C source into pycparser AST."""

from __future__ import annotations

from pathlib import Path

from pycparser import CParser, c_ast, parse_file


def parse_c_string(source: str) -> c_ast.FileAST:
    """Parse a C source string into a pycparser AST."""
    parser = CParser()
    return parser.parse(source, filename="<string>")


def parse_c_file(
    path: str | Path, *, cpp_args: list[str] | None = None
) -> c_ast.FileAST:
    """Parse a C file, optionally running the preprocessor.

    If cpp_args is None, uses a fake preprocessor that strips #includes
    and #defines -- suitable for pre-preprocessed files.
    """
    return parse_file(str(path), use_cpp=cpp_args is not None, cpp_args=cpp_args or [])
