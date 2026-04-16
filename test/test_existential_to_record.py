"""Tests for ExistentialToRecord pass.

Lowering snapshots verify the IR after ExistentialToRecord only.
End-to-end tests verify the full JIT pipeline: ASM parse →
lowering → LLVM codegen → JIT execution → result verification.
"""

from __future__ import annotations

import subprocess

import dgen
from dgen.asm.parser import parse
from dgen.dialects import existential
from dgen.dialects.index import Index
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.memory import Memory
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.existential_to_record import ExistentialToRecord
from dgen.passes.record_to_memory import RecordToMemory
from dgen.testing import strip_prefix


def _lower(ir_text: str) -> dgen.Value:
    """Lower through ExistentialToRecord only."""
    return Compiler(
        [ExistentialToRecord()],
        IdentityPass(),
    ).run(parse(ir_text))


def _compile(ir_text: str):
    """Parse ASM and compile through the existential-aware pipeline."""
    return Compiler(
        passes=[
            ControlFlowToGoto(),
            ExistentialToRecord(),
            RecordToMemory(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        exit=LLVMCodegen(),
    ).run(parse(ir_text))


# ---------------------------------------------------------------------------
# Lowering snapshots: verify the IR after ExistentialToRecord only
# ---------------------------------------------------------------------------


def test_lowering_pack(ir_snapshot) -> None:
    """Lowered IR for existential.pack: record.pack + heap_box."""
    assert (
        _lower(
            strip_prefix("""
        | import existential
        | import index
        |
        | %x : index.Index = 42
        | %packed : existential.Some<index.Index> = existential.pack(%x)
    """)
        )
        == ir_snapshot
    )


def test_lowering_unpack(ir_snapshot) -> None:
    """Lowered IR for existential.unpack: memory.load + record.get."""
    assert (
        _lower(
            strip_prefix("""
        | import existential
        | import index
        |
        | %x : index.Index = 42
        | %packed : existential.Some<index.Index> = existential.pack(%x)
        | %result : index.Index = existential.unpack(%packed)
    """)
        )
        == ir_snapshot
    )


# ---------------------------------------------------------------------------
# Pack: function returns an existential
# ---------------------------------------------------------------------------


def test_pack_and_return_existential() -> None:
    """Pack an argument and return the Some. Verify witness AND value."""
    exe = _compile(
        strip_prefix("""
        | import existential
        | import function
        | import index
        |
        | %main : function.Function<[index.Index], existential.Some<index.Index>> = function.function<existential.Some<index.Index>>() body(%x: index.Index):
        |     %packed : existential.Some<index.Index> = existential.pack(%x)
    """)
    )
    payload = exe.run(42).to_json()
    assert payload["type"]["tag"] == "index.Index"
    assert payload["value"] == 42


# ---------------------------------------------------------------------------
# Existential as function argument
# ---------------------------------------------------------------------------


def test_pass_existential_as_argument() -> None:
    """Pass a Some through a function and get it back unchanged."""
    exe = _compile(
        strip_prefix("""
        | import existential
        | import function
        | import index
        |
        | %main : function.Function<[existential.Some<index.Index>], existential.Some<index.Index>> = function.function<existential.Some<index.Index>>() body(%box: existential.Some<index.Index>):
        |     %_ : existential.Some<index.Index> = chain(%box, ())
    """)
    )
    some_index = existential.Some(bound=Index())
    input_payload = {"type": Index().to_json(), "value": 7}
    mem = Memory.from_json(some_index, input_payload)
    result = exe.run(mem)
    assert result.to_json() == input_payload


# ---------------------------------------------------------------------------
# Full pack → unpack round-trip
# ---------------------------------------------------------------------------


def test_pack_unpack_roundtrip() -> None:
    """Pack then unpack: the original value survives the existential cycle."""
    exe = _compile(
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
    assert exe.run(42).to_json() == 42


def test_pack_unpack_roundtrip_different_values() -> None:
    """Pack/unpack preserves different values, not just one."""
    exe = _compile(
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
    for v in [0, 1, -1, 42, 999, 2**32]:
        assert exe.run(v).to_json() == v


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_some_to_json_preserves_value() -> None:
    """A packed Some's to_json includes the witness type AND the actual value."""
    exe = _compile(
        strip_prefix("""
        | import existential
        | import function
        | import index
        |
        | %main : function.Function<[index.Index], existential.Some<index.Index>> = function.function<existential.Some<index.Index>>() body(%x: index.Index):
        |     %packed : existential.Some<index.Index> = existential.pack(%x)
    """)
    )
    assert exe.run(42).to_json() == {
        "type": {"tag": "index.Index", "params": {}},
        "value": 42,
    }


def test_some_to_native_value_preserves_value() -> None:
    """The rich path rehydrates the witness to a Type instance."""
    exe = _compile(
        strip_prefix("""
        | import existential
        | import function
        | import index
        |
        | %main : function.Function<[index.Index], existential.Some<index.Index>> = function.function<existential.Some<index.Index>>() body(%x: index.Index):
        |     %packed : existential.Some<index.Index> = existential.pack(%x)
    """)
    )
    rich = exe.run(42).to_native_value()
    assert isinstance(rich["type"], Index)
    assert rich["value"] == 42


def test_some_from_json_roundtrip() -> None:
    """Memory.from_json with a value round-trips through to_json."""
    some_index = existential.Some(bound=Index())
    payload = {"type": Index().to_json(), "value": 42}
    mem = Memory.from_json(some_index, payload)
    assert mem.to_json() == payload


def test_cli_shows_packed_value() -> None:
    """python -m dgen on a constant pack shows the witness and value in ASM format."""
    result = subprocess.run(
        ["python", "-m", "dgen", "examples/dependent_types/existential_any.dgen.asm"],
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    assert "number.SignedInteger<index.Index(32)>" in output
    assert '"value": 4' in output
