"""Tests for Python .pyi stub generator (module introspection based)."""

from __future__ import annotations

import importlib
import sys

from dgen.gen.importer import DgenLoader
from dgen.gen.python import generate_pyi


# ---------------------------------------------------------------------------
# Builtin dialect tests
# ---------------------------------------------------------------------------


def test_generate_builtin_header():
    mod = importlib.import_module("dgen.dialects.builtin")
    code = generate_pyi(mod, "builtin")
    assert "# GENERATED" in code
    assert "from dgen import" in code
    assert 'Dialect("builtin")' in code


def test_generate_builtin_no_trait():
    mod = importlib.import_module("dgen.dialects.builtin")
    code = generate_pyi(mod, "builtin")
    assert "HasSingleBlock" not in code


def test_generate_number_simple_type():
    mod = importlib.import_module("dgen.dialects.number")
    code = generate_pyi(mod, "number")
    assert "@dataclass(frozen=True, eq=False)" in code
    assert "class Float64(Type):" in code


def test_generate_builtin_parametric_type():
    mod = importlib.import_module("dgen.dialects.builtin")
    code = generate_pyi(mod, "builtin")
    assert "class Array(Type):" in code
    assert "element_type: Value[dgen.TypeType]" in code
    assert "n: Value[Index]" in code


def test_generate_builtin_span_type():
    mod = importlib.import_module("dgen.dialects.builtin")
    code = generate_pyi(mod, "builtin")
    assert "class Tuple(Type):" in code
    assert "types: list[Value[dgen.TypeType]]" in code


def test_generate_algebra_op():
    mod = importlib.import_module("dgen.dialects.algebra")
    code = generate_pyi(mod, "algebra")
    assert "class AddOp(Op):" in code
    assert "type: Type" in code


def test_generate_builtin_op_with_optional_operand():
    mod = importlib.import_module("dgen.dialects.builtin")
    code = generate_pyi(mod, "builtin")
    assert "class ChainOp(Op):" in code
    assert "lhs: Value" in code


def test_generate_builtin_op_kw_only_decorator():
    mod = importlib.import_module("dgen.dialects.builtin")
    code = generate_pyi(mod, "builtin")
    assert "@dataclass(eq=False, kw_only=True)" in code


# ---------------------------------------------------------------------------
# Function dialect tests
# ---------------------------------------------------------------------------


def test_generate_function_type():
    """Function<Type> should NOT have list annotation (it's not a list container)."""
    mod = importlib.import_module("dgen.dialects.function")
    code = generate_pyi(mod, "function")
    assert "class Function(Type):" in code
    assert "result: Value[dgen.TypeType]" in code


def test_generate_function_define_op_with_block():
    mod = importlib.import_module("dgen.dialects.function")
    code = generate_pyi(mod, "function")
    assert "class FunctionOp(Op):" in code
    assert "body: Block" in code
    assert "Block" in code.split("from dgen import")[1].split("\n")[0]


def test_generate_function_call_op():
    """CallOp.callee should be Value[Function], not list[Value[...]]."""
    mod = importlib.import_module("dgen.dialects.function")
    code = generate_pyi(mod, "function")
    assert "callee: Value[Function]" in code


def test_generate_function_valid_python():
    mod = importlib.import_module("dgen.dialects.function")
    code = generate_pyi(mod, "function")
    compile(code, "<function.pyi>", "exec")


def test_generate_builtin_valid_python():
    mod = importlib.import_module("dgen.dialects.builtin")
    code = generate_pyi(mod, "builtin")
    compile(code, "<builtin.pyi>", "exec")


# ---------------------------------------------------------------------------
# LLVM dialect tests
# ---------------------------------------------------------------------------


def test_generate_llvm_parameterized_default():
    """Int(bits=Index().constant(64)) should appear as a readable default."""
    mod = importlib.import_module("dgen.dialects.llvm")
    code = generate_pyi(mod, "llvm")
    assert "type: Type = Int(bits=Index().constant(64))" in code


def test_generate_llvm_imports():
    mod = importlib.import_module("dgen.dialects.llvm")
    code = generate_pyi(mod, "llvm")
    assert "from dgen.dialects.builtin import" in code
    assert "Index" in code


def test_generate_llvm_valid_python():
    mod = importlib.import_module("dgen.dialects.llvm")
    code = generate_pyi(mod, "llvm")
    compile(code, "<llvm.pyi>", "exec")


# ---------------------------------------------------------------------------
# Control flow dialect tests
# ---------------------------------------------------------------------------


def test_generate_control_flow_op_trait_base():
    mod = importlib.import_module("dgen.dialects.control_flow")
    code = generate_pyi(mod, "control_flow")
    assert "class ForOp(Op):" in code


# ---------------------------------------------------------------------------
# Toy dialect tests
# ---------------------------------------------------------------------------


def test_generate_toy_module_alias_import():
    """Module alias import (import ndbuffer) should appear in stub."""
    mod = importlib.import_module("toy.dialects.toy")
    code = generate_pyi(mod, "toy")
    assert "import dgen.dialects.ndbuffer as ndbuffer" in code


def test_generate_toy_cross_dialect_param():
    """Tensor.shape should reference ndbuffer.Shape via the module alias."""
    mod = importlib.import_module("toy.dialects.toy")
    code = generate_pyi(mod, "toy")
    assert "shape: Value[ndbuffer.Shape]" in code


def test_generate_toy_default_param():
    mod = importlib.import_module("toy.dialects.toy")
    code = generate_pyi(mod, "toy")
    assert "dtype: Value[dgen.TypeType] = Float64()" in code


def test_generate_toy_valid_python():
    mod = importlib.import_module("toy.dialects.toy")
    code = generate_pyi(mod, "toy")
    compile(code, "<toy.pyi>", "exec")


# ---------------------------------------------------------------------------
# Import hook tests (unchanged from before)
# ---------------------------------------------------------------------------


def test_import_hook_loads_builtin():
    """The .dgen import hook makes dgen.dialects.builtin importable."""
    mod = importlib.import_module("dgen.dialects.builtin")
    assert hasattr(mod, "Index")
    assert hasattr(mod, "builtin")
    number_mod = importlib.import_module("dgen.dialects.number")
    assert hasattr(number_mod, "Float64")


def test_import_hook_loads_toy():
    """The .dgen import hook resolves cross-file imports (ndbuffer in toy)."""
    mod = importlib.import_module("toy.dialects.toy")
    assert hasattr(mod, "Tensor")
    assert hasattr(mod, "toy")


def test_import_hook_import_map_auto_resolved():
    """DgenLoader stores the resolved import_map after loading."""
    spec = sys.modules["dgen.dialects.builtin"].__spec__
    assert isinstance(spec.loader, DgenLoader)
    # builtin.dgen imports index dialect
    assert spec.loader.import_map == {"index": "dgen.dialects.index"}


def test_import_hook_toy_import_map_has_ndbuffer():
    """Loader for toy.dgen resolves 'ndbuffer' → 'dgen.dialects.ndbuffer'."""
    spec = sys.modules["toy.dialects.toy"].__spec__
    assert isinstance(spec.loader, DgenLoader)
    assert spec.loader.import_map.get("ndbuffer") == "dgen.dialects.ndbuffer"
