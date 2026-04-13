"""Test that Compiler[Value] can be used as a Pass inside another Compiler."""

from __future__ import annotations

from dgen.asm.parser import parse
from dgen.llvm import lower_to_llvm
from dgen.passes import lower_builtin_dialects
from dgen.passes.compiler import Compiler
from dgen.testing import strip_prefix


def test_compiler_as_pass_in_pipeline() -> None:
    """A Compiler[Value] works as a pass inside another compiler."""
    outer = Compiler(
        passes=[lower_builtin_dialects()],
        exit=lower_to_llvm(),
    )

    exe = outer.run(
        parse(
            strip_prefix("""
        | import function
        | import index
        | import record
        |
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %packed : Span<index.Index> = record.pack([%a, %b])
        |     %result : index.Index = record.get<index.Index(1)>(%packed)
    """)
        )
    )
    assert exe.run(10, 20).to_json() == 20


def test_nested_compilers() -> None:
    """Compilers can be nested: lower_builtin_dialects inside lower_to_llvm's pipeline."""
    outer = Compiler(
        passes=[lower_builtin_dialects()],
        exit=lower_to_llvm(),
    )

    exe = outer.run(
        parse(
            strip_prefix("""
        | import existential
        | import function
        | import index
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %packed : existential.Some<index.Index> = existential.pack(%x)
        |     %result : index.Index = existential.unpack(%packed)
    """)
        )
    )
    assert exe.run(42).to_json() == 42
