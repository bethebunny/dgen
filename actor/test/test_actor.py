"""Actor pipeline tests: fusion, JIT, and staging."""

from __future__ import annotations

import pytest

from actor.passes.actor_to_affine import ActorToAffine
from dgen import asm
from dgen.codegen import Executable, LLVMCodegen
from dgen.compiler import Compiler
from dgen.dialects import builtin
from dgen.testing import strip_prefix
from dgen.type import Memory
from toy.passes.affine_to_llvm import AffineToLLVMLowering

actor_compiler: Compiler[Executable] = Compiler(
    passes=[ActorToAffine(), AffineToLLVMLowering()],
    exit=LLVMCodegen(),
)


def _run_actor_pipeline(module: object) -> list[float]:
    """Compile and run an actor pipeline, returning output as a float list.

    MemRef is an opaque pointer (Pointer<Void>) — we pass the raw data
    address to the JIT and read the result pointer directly.
    """
    import ctypes

    exe = actor_compiler.compile(module)  # type: ignore[arg-type]
    f64x4 = builtin.Array(element_type=builtin.F64(), n=builtin.Index().constant(4))
    input_data = Memory.from_value(f64x4, [1.0, 2.0, 3.0, 4.0])

    # JIT expects raw pointer, returns raw pointer
    from dgen.codegen import _jit_engine
    engine = _jit_engine(exe)
    func_ptr = engine.get_function_address(exe.main_name)
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)(func_ptr)
    result_ptr = cfunc(input_data.address)

    return Memory.from_raw(f64x4, result_ptr).to_json()  # type: ignore[return-value]


def test_fused_pipeline() -> None:
    """Two actors with equal rates. input * 2 + 1 per element."""
    module = asm.parse(strip_prefix("""
        | import actor
        | import affine
        |
        | %main : Nil = function<affine.MemRef<affine.Shape<1>([4]), F64>>() body(%0: affine.MemRef<affine.Shape<1>([4]), F64>):
        |     %1 : affine.MemRef<affine.Shape<1>([4]), F64> = actor.pipeline(%0) body(%2: affine.MemRef<affine.Shape<1>([4]), F64>):
        |         %3 : Nil = actor.actor<4, 4>(%2) body(%4: affine.MemRef<affine.Shape<1>([4]), F64>):
        |             %5 : affine.MemRef<affine.Shape<1>([4]), F64> = affine.alloc(affine.Shape<1>([4]))
        |             %6 : Nil = affine.for<0, 4>([%4, %5]) body(%7: Index, %input: affine.MemRef<affine.Shape<1>([4]), F64>, %out: affine.MemRef<affine.Shape<1>([4]), F64>):
        |                 %8 : F64 = affine.load(%input, [%7])
        |                 %9 : F64 = 2.0
        |                 %10 : F64 = affine.mul_f(%8, %9)
        |                 %11 : Nil = affine.store(%10, %out, [%7])
        |             %12 : affine.MemRef<affine.Shape<1>([4]), F64> = chain(%5, %6)
        |             %13 : Nil = actor.produce(%12)
        |         %14 : Nil = actor.actor<4, 4>(%3) body(%15: affine.MemRef<affine.Shape<1>([4]), F64>):
        |             %16 : affine.MemRef<affine.Shape<1>([4]), F64> = affine.alloc(affine.Shape<1>([4]))
        |             %17 : Nil = affine.for<0, 4>([%15, %16]) body(%18: Index, %input: affine.MemRef<affine.Shape<1>([4]), F64>, %out: affine.MemRef<affine.Shape<1>([4]), F64>):
        |                 %19 : F64 = affine.load(%input, [%18])
        |                 %20 : F64 = 1.0
        |                 %21 : F64 = affine.add_f(%19, %20)
        |                 %22 : Nil = affine.store(%21, %out, [%18])
        |             %23 : affine.MemRef<affine.Shape<1>([4]), F64> = chain(%16, %17)
        |             %24 : Nil = actor.produce(%23)
    """))
    assert _run_actor_pipeline(module) == [3.0, 5.0, 7.0, 9.0]


def test_unfused_pipeline() -> None:
    """Two actors with different rates. input * 2, then first 2 elements + 1."""
    import ctypes

    module = asm.parse(strip_prefix("""
        | import actor
        | import affine
        |
        | %main : Nil = function<affine.MemRef<affine.Shape<1>([4]), F64>>() body(%0: affine.MemRef<affine.Shape<1>([4]), F64>):
        |     %1 : affine.MemRef<affine.Shape<1>([2]), F64> = actor.pipeline(%0) body(%2: affine.MemRef<affine.Shape<1>([4]), F64>):
        |         %3 : Nil = actor.actor<4, 4>(%2) body(%4: affine.MemRef<affine.Shape<1>([4]), F64>):
        |             %5 : affine.MemRef<affine.Shape<1>([4]), F64> = affine.alloc(affine.Shape<1>([4]))
        |             %6 : Nil = affine.for<0, 4>([%4, %5]) body(%7: Index, %input: affine.MemRef<affine.Shape<1>([4]), F64>, %out: affine.MemRef<affine.Shape<1>([4]), F64>):
        |                 %8 : F64 = affine.load(%input, [%7])
        |                 %9 : F64 = 2.0
        |                 %10 : F64 = affine.mul_f(%8, %9)
        |                 %11 : Nil = affine.store(%10, %out, [%7])
        |             %12 : affine.MemRef<affine.Shape<1>([4]), F64> = chain(%5, %6)
        |             %13 : Nil = actor.produce(%12)
        |         %14 : Nil = actor.actor<2, 2>(%3) body(%15: affine.MemRef<affine.Shape<1>([4]), F64>):
        |             %16 : affine.MemRef<affine.Shape<1>([2]), F64> = affine.alloc(affine.Shape<1>([2]))
        |             %17 : Nil = affine.for<0, 2>([%15, %16]) body(%18: Index, %input: affine.MemRef<affine.Shape<1>([4]), F64>, %out: affine.MemRef<affine.Shape<1>([2]), F64>):
        |                 %19 : F64 = affine.load(%input, [%18])
        |                 %20 : F64 = 1.0
        |                 %21 : F64 = affine.add_f(%19, %20)
        |                 %22 : Nil = affine.store(%21, %out, [%18])
        |             %23 : affine.MemRef<affine.Shape<1>([2]), F64> = chain(%16, %17)
        |             %24 : Nil = actor.produce(%23)
    """))
    exe = actor_compiler.compile(module)
    f64x4 = builtin.Array(element_type=builtin.F64(), n=builtin.Index().constant(4))
    f64x2 = builtin.Array(element_type=builtin.F64(), n=builtin.Index().constant(2))
    input_data = Memory.from_value(f64x4, [1.0, 2.0, 3.0, 4.0])

    from dgen.codegen import _jit_engine
    engine = _jit_engine(exe)
    func_ptr = engine.get_function_address(exe.main_name)
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)(func_ptr)
    result_ptr = cfunc(input_data.address)
    assert Memory.from_raw(f64x2, result_ptr).to_json() == [3.0, 5.0]


def test_lowering_ir(ir_snapshot: object) -> None:
    """Pipeline lowers to inlined actor bodies."""
    module = asm.parse(strip_prefix("""
        | import actor
        | import affine
        |
        | %main : Nil = function<affine.MemRef<affine.Shape<1>([4]), F64>>() body(%0: affine.MemRef<affine.Shape<1>([4]), F64>):
        |     %1 : affine.MemRef<affine.Shape<1>([4]), F64> = actor.pipeline(%0) body(%2: affine.MemRef<affine.Shape<1>([4]), F64>):
        |         %3 : Nil = actor.actor<4, 4>(%2) body(%4: affine.MemRef<affine.Shape<1>([4]), F64>):
        |             %5 : affine.MemRef<affine.Shape<1>([4]), F64> = affine.alloc(affine.Shape<1>([4]))
        |             %6 : Nil = affine.for<0, 4>([%4, %5]) body(%7: Index, %input: affine.MemRef<affine.Shape<1>([4]), F64>, %out: affine.MemRef<affine.Shape<1>([4]), F64>):
        |                 %8 : F64 = affine.load(%input, [%7])
        |                 %9 : F64 = 2.0
        |                 %10 : F64 = affine.mul_f(%8, %9)
        |                 %11 : Nil = affine.store(%10, %out, [%7])
        |             %12 : affine.MemRef<affine.Shape<1>([4]), F64> = chain(%5, %6)
        |             %13 : Nil = actor.produce(%12)
        |         %14 : Nil = actor.actor<4, 4>(%3) body(%15: affine.MemRef<affine.Shape<1>([4]), F64>):
        |             %16 : affine.MemRef<affine.Shape<1>([4]), F64> = affine.alloc(affine.Shape<1>([4]))
        |             %17 : Nil = affine.for<0, 4>([%15, %16]) body(%18: Index, %input: affine.MemRef<affine.Shape<1>([4]), F64>, %out: affine.MemRef<affine.Shape<1>([4]), F64>):
        |                 %19 : F64 = affine.load(%input, [%18])
        |                 %20 : F64 = 1.0
        |                 %21 : F64 = affine.add_f(%19, %20)
        |                 %22 : Nil = affine.store(%21, %out, [%18])
        |             %23 : affine.MemRef<affine.Shape<1>([4]), F64> = chain(%16, %17)
        |             %24 : Nil = actor.produce(%23)
    """))
    lowered = Compiler(passes=[ActorToAffine()], exit=LLVMCodegen()).run(module)
    assert lowered == ir_snapshot
