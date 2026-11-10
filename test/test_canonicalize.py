"""AlgebraCanonicalize: constant folding and identities, to a fixed point."""

import pytest

from dgen import asm
from dgen.asm.parser import parse
from dgen.compiler import Compiler, IdentityPass
from dgen.passes.canonicalize import AlgebraCanonicalize
from dgen.testing import llvm_compile, strip_prefix


def canonical(body: str, *, args: str = "%x: index.Index") -> str:
    ir = strip_prefix(f"""
        | import algebra
        | import function
        | import index
        | import number
        |
        | %f : function.Function<[index.Index], index.Index> = function.function<index.Index>() body({args}):
    """) + "\n".join(f"    {line}" for line in body.strip().splitlines())
    lowered = Compiler([AlgebraCanonicalize()], IdentityPass()).run(parse(ir))
    return asm.format(lowered)


def test_folds_constants():
    out = canonical("""
%a : index.Index = 2
%b : index.Index = 3
%c : index.Index = algebra.add(%a, %b)
%d : index.Index = algebra.multiply(%c, %x)
""")
    assert "algebra.add" not in out
    assert "= 5" in out


def test_folds_chains():
    out = canonical("""
%a : index.Index = 2
%b : index.Index = 3
%c : index.Index = algebra.multiply(%a, %b)
%d : index.Index = algebra.subtract(%c, %a)
""")
    assert "algebra" not in out
    assert "= 4" in out


def test_add_zero_is_identity():
    out = canonical("""
%zero : index.Index = 0
%y : index.Index = algebra.add(%x, %zero)
""")
    assert "algebra" not in out


def test_commuted_constant_folds_on_the_next_walk():
    # add(0, x) becomes add(x, 0), a new op, which only a fixed point folds.
    out = canonical("""
%zero : index.Index = 0
%one : index.Index = 1
%y : index.Index = algebra.add(%zero, %x)
%z : index.Index = algebra.multiply(%one, %y)
""")
    assert "algebra" not in out


def test_subtract_self_is_zero():
    out = canonical("""
%y : index.Index = algebra.subtract(%x, %x)
""")
    assert "algebra" not in out


def test_double_negate():
    out = canonical("""
%n : index.Index = algebra.negate(%x)
%y : index.Index = algebra.negate(%n)
""")
    assert "algebra" not in out


def test_folds_comparison():
    out = canonical("""
%a : index.Index = 2
%b : index.Index = 3
%c : number.Boolean = algebra.less_than(%a, %b)
%y : index.Index = algebra.add(%x, %x)
""")
    assert "less_than" not in out


def test_division_by_zero_is_left():
    out = canonical("""
%a : index.Index = 7
%zero : index.Index = 0
%y : index.Index = algebra.divide(%a, %zero)
""")
    assert "algebra.divide" in out


@pytest.mark.parametrize(
    ("left", "right", "quotient"), [(7, 2, 3), (-7, 2, -3), (7, -2, -3)]
)
def test_integer_division_truncates(left: int, right: int, quotient: int):
    out = canonical(f"""
%a : index.Index = {left}
%b : index.Index = {right}
%y : index.Index = algebra.divide(%a, %b)
""")
    assert f"= {quotient}" in out


def test_folds_with_64_bit_wrap():
    out = canonical("""
%a : index.Index = 9223372036854775807
%one : index.Index = 1
%y : index.Index = algebra.add(%a, %one)
""")
    assert "= -9223372036854775808" in out


def test_folded_function_runs():
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        |
        | %f : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %zero : index.Index = 0
        |     %two : index.Index = 2
        |     %three : index.Index = 3
        |     %six : index.Index = algebra.multiply(%two, %three)
        |     %y : index.Index = algebra.add(%zero, %x)
        |     %z : index.Index = algebra.add(%y, %six)
    """)
    lowered = Compiler([AlgebraCanonicalize()], IdentityPass()).run(parse(ir))
    assert llvm_compile(lowered).run(10).to_json() == 16
