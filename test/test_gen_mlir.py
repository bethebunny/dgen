"""Tests for the MLIR ODS generator.

The snapshots in test/mlir are the generator's output. After a deliberate change, regenerate one with
    python -m dgen.gen --target mlir dgen/dialects/X.dgen > test/mlir/X.td
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from dgen.gen import model
from dgen.gen.importer import model_of
from dgen.gen.mlir import generate_td
from dgen.gen.parser import parse

SNAPSHOTS = Path(__file__).parent / "mlir"
PRELUDE = Path(__file__).parents[1] / "dgen" / "gen"
DIALECTS = {
    "index": "dgen.dialects.index",
    "builtin": "dgen.dialects.builtin",
    "number": "dgen.dialects.number",
    "algebra": "dgen.dialects.algebra",
    "control_flow": "dgen.dialects.control_flow",
}


def td(module_name: str) -> str:
    return generate_td(model_of(importlib.import_module(module_name)))


def framework_dialect(name: str) -> model.Dialect:
    return model_of(importlib.import_module(f"dgen.dialects.{name}"))


def td_of(source: str) -> str:
    return generate_td(
        model.Dialect.from_ast(
            parse(source), name="t", module="t", imported=framework_dialect
        )
    )


@pytest.mark.parametrize("name", DIALECTS)
def test_snapshot(name):
    assert td(DIALECTS[name]) == (SNAPSHOTS / f"{name}.td").read_text()


def test_dialect_name_takes_prefix():
    text = generate_td(
        model_of(importlib.import_module("dgen.dialects.index")), prefix="my_"
    )
    assert 'let name = "my_index";' in text


def test_includes_referenced_dialects():
    text = td("dgen.dialects.control_flow")
    assert 'include "builtin.td"' in text
    assert 'include "index.td"' in text


def test_value_parameter_is_typed_attribute():
    text = td("dgen.dialects.control_flow")
    assert "Dgen_ValueAttr<Index_Index>:$lower_bound" in text


def test_type_parameter_is_type_attribute():
    text = td_of("op f<t: Type>()\n")
    assert "TypeAttr:$t" in text


def test_block_is_single_block_region():
    text = td("dgen.dialects.control_flow")
    assert "SizedRegion<1>:$then_body" in text


def test_type_result():
    assert "(outs Dgen_Type:$result)" in td("dgen.dialects.builtin")


def test_type_parameter_default():
    text = td_of(
        "from number import Float64\ntype T<dtype: Type = Float64>:\n    layout Int\n"
    )
    assert (
        'DefaultValuedParameter<"::mlir::Type", "::dgen::number::Float64Type::get($_ctxt)">:$dtype'
        in text
    )
    assert "`<` ($dtype^)? `>`" in text


TRAITS = """\
trait Numeric
type Num:
    layout Int
    has trait Numeric
op negate(input) -> Num:
    has trait Numeric
    requires input has trait Numeric
op scale<k: Num>(input) -> Num:
    requires k < 4
"""


def test_traits_and_constraints():
    text = td_of(TRAITS)
    assert 'def T_NumericTypeInterface : TypeInterface<"NumericTypeInterface">' in text
    assert 'TypeDef<T_Dialect, "Num", [T_NumericTypeInterface]>' in text
    assert (
        'T_Op<"negate", [T_NumericOpInterface, TypeIs<"input", T_NumericTypeInterface>]>'
        in text
    )
    assert "// dgen: requires k < 4 has no ODS form" in text


def test_dropped_type_arguments_are_noted():
    text = td_of(
        "from index import Index\nfrom builtin import Array\nop f(a: Array<Index, 4>)\n"
    )
    assert (
        "// dgen: operand a is builtin.Array<index.Index, 4>, checked as Builtin_Array only"
        in text
    )


@pytest.mark.slow
def test_td_compiles(tmp_path):
    """Runs mlir-tblgen over every snapshot, with -m slow and MLIR_INCLUDE set to MLIR's include dir."""
    tblgen = shutil.which("mlir-tblgen")
    include = os.environ.get("MLIR_INCLUDE")
    if tblgen is None or include is None:
        pytest.skip("needs mlir-tblgen on the path and MLIR_INCLUDE")
    for name in DIALECTS:
        for generator in ["-gen-typedef-decls", "-gen-op-decls"]:
            subprocess.run(
                [
                    tblgen,
                    generator,
                    f"-I{include}",
                    f"-I{PRELUDE}",
                    f"-I{SNAPSHOTS}",
                    str(SNAPSHOTS / f"{name}.td"),
                    "-o",
                    str(tmp_path / f"{name}{generator}.inc"),
                ],
                check=True,
            )
