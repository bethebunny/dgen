"""Actor pipeline tests: fusion, JIT, and staging."""

from __future__ import annotations

import pytest

from actor.passes.actor_to_affine import ActorToAffine
from dgen import asm
from dgen.codegen import Executable, LLVMCodegen
from dgen.compiler import Compiler
from dgen.dialects import builtin
from dgen.dialects.number import Float64
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.testing import strip_prefix
from dgen.type import Memory
from toy.passes.control_flow_to_goto import ControlFlowToGoto
from toy.passes.memory_to_llvm import MemoryToLLVM
from toy.passes.ndbuffer_to_memory import NDBufferToMemory

actor_compiler: Compiler[Executable] = Compiler(
    passes=[
        ActorToAffine(),
        ControlFlowToGoto(),
        NDBufferToMemory(),
        MemoryToLLVM(),
        AlgebraToLLVM(),
    ],
    exit=LLVMCodegen(),
)

# Shared IR template: two actors, multiply-by-2 then add-1.
# For-loop bodies have explicit capture via captures to satisfy
# the closed-block invariant.
_PIPELINE_IR = strip_prefix("""
    | import actor
    | import algebra
    | import control_flow
    | import function
    | import ndbuffer
    | import number
    |
    | %main : function.Function<ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>> = function.function<ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>>() body(%0: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
    |     %1 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64> = actor.pipeline(%0) body(%2: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
    |         %3 : Nil = actor.actor<4, 4>(%2) body(%4: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
    |             %5 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<1>([4]))
    |             %6 : Nil = control_flow.for<0, 4>([%4, %5]) body(%7: index.Index, %input: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>, %out: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
    |                 %8 : number.Float64 = ndbuffer.load(%input, [%7])
    |                 %9 : number.Float64 = 2.0
    |                 %10 : number.Float64 = algebra.multiply(%8, %9)
    |                 %11 : Nil = ndbuffer.store(%10, %out, [%7])
    |             %12 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64> = chain(%5, %6)
    |             %13 : Nil = actor.produce(%12)
    |         %14 : Nil = actor.actor<4, 4>(%3) body(%15: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
    |             %16 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<1>([4]))
    |             %17 : Nil = control_flow.for<0, 4>([%15, %16]) body(%18: index.Index, %input: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>, %out: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
    |                 %19 : number.Float64 = ndbuffer.load(%input, [%18])
    |                 %20 : number.Float64 = 1.0
    |                 %21 : number.Float64 = algebra.add(%19, %20)
    |                 %22 : Nil = ndbuffer.store(%21, %out, [%18])
    |             %23 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64> = chain(%16, %17)
    |             %24 : Nil = actor.produce(%23)
""")


@pytest.mark.xfail(reason="JIT malloc return pointer read-back needs investigation")
def test_fused_pipeline() -> None:
    """Two actors with equal rates. input * 2 + 1 per element."""
    module = asm.parse(_PIPELINE_IR)
    exe = actor_compiler.compile(module)
    memref = exe.input_types[0]
    f64x4 = builtin.Array(element_type=Float64(), n=builtin.Index().constant(4))
    input_data = Memory.from_value(f64x4, [1.0, 2.0, 3.0, 4.0])
    input_mem = Memory.from_value(memref, input_data.address)
    result = exe.run(input_mem)
    ptr = result.unpack()[0]
    assert Memory.from_raw(f64x4, ptr).to_json() == [3.0, 5.0, 7.0, 9.0]


@pytest.mark.xfail(reason="JIT malloc return pointer read-back needs investigation")
def test_unfused_pipeline() -> None:
    """Two actors with different rates. input * 2, then first 2 elements + 1."""
    module = asm.parse(strip_prefix("""
        | import actor
        | import algebra
        | import control_flow
        | import function
        | import ndbuffer
        | import number
        |
        | %main : function.Function<ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>> = function.function<ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>>() body(%0: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
        |     %1 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([2]), number.Float64> = actor.pipeline(%0) body(%2: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
        |         %3 : Nil = actor.actor<4, 4>(%2) body(%4: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
        |             %5 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<1>([4]))
        |             %6 : Nil = control_flow.for<0, 4>([%4, %5]) body(%7: index.Index, %input: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>, %out: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
        |                 %8 : number.Float64 = ndbuffer.load(%input, [%7])
        |                 %9 : number.Float64 = 2.0
        |                 %10 : number.Float64 = algebra.multiply(%8, %9)
        |                 %11 : Nil = ndbuffer.store(%10, %out, [%7])
        |             %12 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64> = chain(%5, %6)
        |             %13 : Nil = actor.produce(%12)
        |         %14 : Nil = actor.actor<2, 2>(%3) body(%15: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>):
        |             %16 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([2]), number.Float64> = ndbuffer.alloc(ndbuffer.Shape<1>([2]))
        |             %17 : Nil = control_flow.for<0, 2>([%15, %16]) body(%18: index.Index, %input: ndbuffer.NDBuffer<ndbuffer.Shape<1>([4]), number.Float64>, %out: ndbuffer.NDBuffer<ndbuffer.Shape<1>([2]), number.Float64>):
        |                 %19 : number.Float64 = ndbuffer.load(%input, [%18])
        |                 %20 : number.Float64 = 1.0
        |                 %21 : number.Float64 = algebra.add(%19, %20)
        |                 %22 : Nil = ndbuffer.store(%21, %out, [%18])
        |             %23 : ndbuffer.NDBuffer<ndbuffer.Shape<1>([2]), number.Float64> = chain(%16, %17)
        |             %24 : Nil = actor.produce(%23)
    """))
    exe = actor_compiler.compile(module)
    memref = exe.input_types[0]
    f64x4 = builtin.Array(element_type=Float64(), n=builtin.Index().constant(4))
    f64x2 = builtin.Array(element_type=Float64(), n=builtin.Index().constant(2))
    input_data = Memory.from_value(f64x4, [1.0, 2.0, 3.0, 4.0])
    input_mem = Memory.from_value(memref, input_data.address)
    result = exe.run(input_mem)
    ptr = result.unpack()[0]
    assert Memory.from_raw(f64x2, ptr).to_json() == [3.0, 5.0]


def test_lowering_ir(ir_snapshot: object) -> None:
    """Pipeline lowers to inlined actor bodies."""
    module = asm.parse(_PIPELINE_IR)
    lowered = Compiler(passes=[ActorToAffine()], exit=LLVMCodegen()).run(module)
    assert lowered == ir_snapshot
