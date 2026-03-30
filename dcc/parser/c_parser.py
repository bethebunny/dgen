"""Parse C source into pycparser AST."""

from __future__ import annotations

from pathlib import Path

from pycparser import CParser, parse_file, c_ast


def parse_c_string(source: str) -> c_ast.FileAST:
    """Parse a C source string into a pycparser AST."""
    parser = CParser()
    return parser.parse(source, filename="<string>")


def parse_c_file(
    path: str | Path, *, cpp_args: list[str] | None = None
) -> c_ast.FileAST:
    """Parse a C file, optionally running the preprocessor.

    If cpp_args is None, uses a fake preprocessor that strips #includes
    and #defines — suitable for pre-preprocessed files like sqlite3.c.
    """
    path = str(path)
    if cpp_args is not None:
        return parse_file(path, use_cpp=True, cpp_args=cpp_args)
    # Use fake preprocessor for pre-preprocessed input
    return parse_file(path, use_cpp=False)
