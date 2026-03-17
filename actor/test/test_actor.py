"""Actor pipeline tests: fusion, JIT, and staging."""

from __future__ import annotations

import ctypes

from dgen import asm
from dgen.codegen import Executable, LLVMCodegen, _jit_engine
from dgen.compiler import Compiler
from dgen.testing import strip_prefix
from toy.passes.affine_to_llvm import AffineToLLVMLowering

from actor.passes.actor_to_affine import ActorToAffine

actor_compiler: Compiler[Executable] = Compiler(
    passes=[ActorToAffine(), AffineToLLVMLowering()],
    exit=LLVMCodegen(),
)


def _make_input(values: list[float]) -> ctypes.Array[ctypes.c_double]:
    """Create a ctypes double array from a list of floats."""
    return (ctypes.c_double * len(values))(*values)


def _read_doubles(ptr: int, n: int) -> list[float]:
    """Read n doubles from a raw pointer returned by JIT."""
    return list((ctypes.c_double * n).from_address(ptr))


def _run_pipeline(exe: Executable, values: list[float], n_out: int) -> list[float]:
    """Run a compiled pipeline: pass input as raw pointer, read output doubles."""
    buf = _make_input(values)
    engine = _jit_engine(exe)
    func_ptr = engine.get_function_address(exe.main_name)
    cfunc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(func_ptr)
    raw = cfunc(ctypes.cast(buf, ctypes.c_void_p).value)
    return _read_doubles(raw, n_out)


def test_fused_pipeline() -> None:
    """rate1 == rate2: fused pipeline (one loop). input * 2 + 1 per element."""
    module = asm.parse(strip_prefix("""
        | import actor
        | import affine
        |
        | %main : Nil = function<affine.MemRef<affine.Shape<1>([4]), F64>>() (%0: affine.MemRef<affine.Shape<1>([4]), F64>):
        |     %1 : affine.MemRef<affine.Shape<1>([4]), F64> = actor.pipeline(%0) (%2: affine.MemRef<affine.Shape<1>([4]), F64>):
        |         %3 : Nil = actor.actor<4, 4>(%2) (%4: F64):
        |             %5 : F64 = 2.0
        |             %6 : F64 = affine.mul_f(%4, %5)
        |             %7 : Nil = actor.produce(%6)
        |         %8 : Nil = actor.actor<4, 4>(%3) (%9: F64):
        |             %10 : F64 = 1.0
        |             %11 : F64 = affine.add_f(%9, %10)
        |             %12 : Nil = actor.produce(%11)
        |     %13 : Nil = return(%1)
    """))
    exe = actor_compiler.compile(module)
    result = _run_pipeline(exe, [1.0, 2.0, 3.0, 4.0], 4)
    assert result == [3.0, 5.0, 7.0, 9.0]


def test_unfused_pipeline() -> None:
    """rate1 != rate2: unfused pipeline (two loops + buffer)."""
    module = asm.parse(strip_prefix("""
        | import actor
        | import affine
        |
        | %main : Nil = function<affine.MemRef<affine.Shape<1>([4]), F64>>() (%0: affine.MemRef<affine.Shape<1>([4]), F64>):
        |     %1 : affine.MemRef<affine.Shape<1>([4]), F64> = actor.pipeline(%0) (%2: affine.MemRef<affine.Shape<1>([4]), F64>):
        |         %3 : Nil = actor.actor<4, 4>(%2) (%4: F64):
        |             %5 : F64 = 2.0
        |             %6 : F64 = affine.mul_f(%4, %5)
        |             %7 : Nil = actor.produce(%6)
        |         %8 : Nil = actor.actor<2, 2>(%3) (%9: F64):
        |             %10 : F64 = 1.0
        |             %11 : F64 = affine.add_f(%9, %10)
        |             %12 : Nil = actor.produce(%11)
        |     %13 : Nil = return(%1)
    """))
    exe = actor_compiler.compile(module)
    result = _run_pipeline(exe, [1.0, 2.0, 3.0, 4.0], 2)
    assert result == [3.0, 5.0]


def test_fused_lowering_ir(ir_snapshot: object) -> None:
    """Fused pipeline lowers to a single affine.for loop."""
    module = asm.parse(strip_prefix("""
        | import actor
        | import affine
        |
        | %main : Nil = function<affine.MemRef<affine.Shape<1>([4]), F64>>() (%0: affine.MemRef<affine.Shape<1>([4]), F64>):
        |     %1 : affine.MemRef<affine.Shape<1>([4]), F64> = actor.pipeline(%0) (%2: affine.MemRef<affine.Shape<1>([4]), F64>):
        |         %3 : Nil = actor.actor<4, 4>(%2) (%4: F64):
        |             %5 : F64 = 2.0
        |             %6 : F64 = affine.mul_f(%4, %5)
        |             %7 : Nil = actor.produce(%6)
        |         %8 : Nil = actor.actor<4, 4>(%3) (%9: F64):
        |             %10 : F64 = 1.0
        |             %11 : F64 = affine.add_f(%9, %10)
        |             %12 : Nil = actor.produce(%11)
        |     %13 : Nil = return(%1)
    """))
    lowered = Compiler(passes=[ActorToAffine()], exit=LLVMCodegen()).run(module)
    assert lowered == ir_snapshot


def test_unfused_lowering_ir(ir_snapshot: object) -> None:
    """Unfused pipeline lowers to two affine.for loops with a buffer."""
    module = asm.parse(strip_prefix("""
        | import actor
        | import affine
        |
        | %main : Nil = function<affine.MemRef<affine.Shape<1>([4]), F64>>() (%0: affine.MemRef<affine.Shape<1>([4]), F64>):
        |     %1 : affine.MemRef<affine.Shape<1>([4]), F64> = actor.pipeline(%0) (%2: affine.MemRef<affine.Shape<1>([4]), F64>):
        |         %3 : Nil = actor.actor<4, 4>(%2) (%4: F64):
        |             %5 : F64 = 2.0
        |             %6 : F64 = affine.mul_f(%4, %5)
        |             %7 : Nil = actor.produce(%6)
        |         %8 : Nil = actor.actor<2, 2>(%3) (%9: F64):
        |             %10 : F64 = 1.0
        |             %11 : F64 = affine.add_f(%9, %10)
        |             %12 : Nil = actor.produce(%11)
        |     %13 : Nil = return(%1)
    """))
    lowered = Compiler(passes=[ActorToAffine()], exit=LLVMCodegen()).run(module)
    assert lowered == ir_snapshot
