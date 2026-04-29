"""Tests for the aggregate-tuple codegen path.

Codegen treats every "list of N values" slot uniformly as a *tuple*: a
``PackOp`` constructed inline emits as an ``insertvalue`` chain producing
a real LLVM aggregate; a runtime aggregate (e.g. a ``Tuple`` returned by
another call) flows through directly. Phi/extractvalue at consumers
materialise individual values.

Tests pin down both shapes end-to-end and snapshot the LLVM IR so the
emitted shape is visible to reviewers.
"""

from __future__ import annotations

from dgen.asm.parser import parse
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.codegen import Executable, LLVMCodegen
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.compiler import Compiler
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.normalize_region_terminators import NormalizeRegionTerminators
from dgen.passes.record_to_memory import RecordToMemory
from dgen.testing import strip_prefix


def _compile(ir_text: str) -> Executable:
    """Pipeline that lowers ``record.pack`` (used to build runtime tuples)
    + the standard control-flow / builtin / algebra lowerings."""
    return Compiler(
        [
            ControlFlowToGoto(),
            NormalizeRegionTerminators(),
            RecordToMemory(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        LLVMCodegen(),
    ).run(parse(ir_text))


# ---------------------------------------------------------------------------
# Inline pack — the common case. ``[%a, %b]`` literal in IR.
# ---------------------------------------------------------------------------


def test_inline_pack_call(snapshot):
    """A call whose arguments are an inline ``[%x, %y]`` PackOp. The
    PackOp emits as an ``insertvalue`` chain; ``extractvalue`` at the
    call site pulls each arg back out. After LLVM opt the round-trip
    folds away."""
    exe = _compile(
        strip_prefix("""
        | import algebra
        | import index
        | import function
        |
        | %add : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %sum : index.Index = algebra.add(%a, %b)
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index, %y: index.Index) captures(%add):
        |     %r : index.Index = function.call<%add>([%x, %y])
    """)
    )
    assert exe.run(11, 31).to_json() == 42
    assert exe.ir == snapshot


# ---------------------------------------------------------------------------
# Runtime tuple as call args — the case that would silently miscompile
# under the old per-arg unpack model.
# ---------------------------------------------------------------------------


def test_call_with_runtime_tuple_as_args(snapshot):
    """A call whose ``arguments`` come from a runtime ``Tuple`` value
    (built here via ``record.pack``, but morally any function returning
    a ``Tuple`` would do). Codegen ``extractvalue``s each LLVM arg from
    the runtime aggregate; the call sees positional scalars."""
    exe = _compile(
        strip_prefix("""
        | import algebra
        | import index
        | import record
        | import function
        |
        | %add : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %sum : index.Index = algebra.add(%a, %b)
        | %main : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index, %y: index.Index) captures(%add):
        |     %t : Tuple<[index.Index, index.Index]> = record.pack([%x, %y])
        |     %r : index.Index = function.call<%add>(%t)
    """)
    )
    assert exe.run(7, 35).to_json() == 42
    assert exe.run(0, 0).to_json() == 0
    assert exe.run(100, 200).to_json() == 300
    assert exe.ir == snapshot


def test_call_with_heterogeneous_tuple_args(snapshot):
    """Mixed-type runtime tuple flowing into a call. ``extractvalue``
    honours each field's individual LLVM type."""
    exe = _compile(
        strip_prefix("""
        | import index
        | import number
        | import record
        | import function
        |
        | %first : function.Function<[index.Index, number.Float64], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: number.Float64):
        |     %_ : index.Index = chain(%a, %b)
        | %main : function.Function<[index.Index, number.Float64], index.Index> = function.function<index.Index>() body(%x: index.Index, %y: number.Float64) captures(%first):
        |     %t : Tuple<[index.Index, number.Float64]> = record.pack([%x, %y])
        |     %r : index.Index = function.call<%first>(%t)
    """)
    )
    assert exe.run(7, 0.5).to_json() == 7
    assert exe.ir == snapshot


# ---------------------------------------------------------------------------
# Runtime pack of TypeType-typed values — the case the prior
# ``_is_type_metadata_pack`` filter would have wrongly silenced. Codegen
# emits the ``insertvalue`` chain because the pack is operand-reachable
# (in ``runtime_dependencies`` of the consumer), regardless of the
# elements' types.
# ---------------------------------------------------------------------------


def test_runtime_pack_of_type_values_is_emitted():
    """A function whose call args are a runtime pack of ``Type`` values
    (here built via ``builtin.type`` of a runtime value). Even though
    the elements are TypeType-typed — the same shape as a ``Tuple``'s
    compile-time ``types`` parameter — the pack is operand-reachable and
    codegen emits its aggregate. A naive "skip TypeType-typed packs"
    filter would silently drop it and produce broken IR.

    Structural-only check (we don't ``exe.run`` because cross-test
    JIT-engine state can shift the runtime Type-descriptor pointers
    that show up inside the inline aggregate). The shape we care about
    is in the emitted text: an ``insertvalue`` chain over
    ``{ ptr, ptr }`` followed by ``extractvalue`` per arg.
    """
    exe = _compile(
        strip_prefix("""
        | import index
        | import function
        |
        | %first_type : function.Function<[Type, Type], Type> = function.function<Type>() body(%a: Type, %b: Type):
        |     %_ : Type = chain(%a, %b)
        | %main : function.Function<[index.Index], Type> = function.function<Type>() body(%x: index.Index) captures(%first_type):
        |     %ta : Type = type(%x)
        |     %r : Type = function.call<%first_type>([%ta, %ta])
    """)
    )
    assert "insertvalue { ptr, ptr }" in exe.ir
    assert "extractvalue { ptr, ptr }" in exe.ir
