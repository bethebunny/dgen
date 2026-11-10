import pytest

import dcc.dialects.dcc
import dcc2.dialects.c
from dgen import Dialect
from dgen.asm.parser import parse
from dgen.dialect import DialectClash
from dgen.dialects.function import FunctionOp
from dgen.testing import strip_prefix


def test_clash_between_origins():
    Dialect("_identity_clash", origin="first.module")
    with pytest.raises(DialectClash, match="from second.module clashes"):
        Dialect("_identity_clash", origin="second.module")


def test_same_origin_replaces():
    first = Dialect("_identity_rebuilt", origin="notebook")
    second = Dialect("_identity_rebuilt", origin="notebook")
    assert first is not second
    assert Dialect.get("_identity_rebuilt") is second


def test_python_dialects_clash():
    Dialect("_identity_python")
    with pytest.raises(DialectClash):
        Dialect("_identity_python")


def test_c_frontends_coexist():
    assert Dialect.get("dcc") is dcc.dialects.dcc.dcc
    assert Dialect.get("c") is dcc2.dialects.c.c


def test_import_names_one_dialect():
    value = parse(
        strip_prefix("""
        | import dcc
        | import function
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : Nil = dcc.return(())
    """)
    )
    assert isinstance(value, FunctionOp)
    assert type(value.body.result) is dcc.dialects.dcc.ReturnOp
