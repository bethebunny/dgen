"""Unit tests for codegen internals: emitter_for, runtime_dependencies, emit_label."""

from dataclasses import dataclass, field

import dgen
from dgen import asm
from dgen.llvm.codegen import (
    EMITTERS,
    _externs,
    emitter_for,
    runtime_dependencies,
    emit,
)
from dgen.dialects import builtin, function, goto, index, llvm
from dgen.builtins import pack
from dgen.testing import strip_prefix

# ---------------------------------------------------------------------------
# emitter_for
# ---------------------------------------------------------------------------


def test_emitter_for_registers_handler():
    """emitter_for(T) registers the decorated function in EMITTERS[T]."""

    class _Dummy:
        pass

    @emitter_for(_Dummy)
    def handle_dummy(op):
        yield "dummy"

    assert EMITTERS[_Dummy] is handle_dummy
    # Clean up so we don't pollute global state for other tests.
    del EMITTERS[_Dummy]


def test_emitter_for_label_registered():
    """goto.LabelOp has a registered emitter (from the @emitter_for decorator at value level)."""
    assert goto.LabelOp in EMITTERS


# ---------------------------------------------------------------------------
# runtime_dependencies
# ---------------------------------------------------------------------------


def _parse(text: str):
    """Parse IR and return the first function's body block."""
    value = asm.parse(strip_prefix(text))
    return value.body


def test_runtime_dependencies_follows_operands():
    """runtime_dependencies yields operand dependencies in topo order."""
    block = _parse("""
        | import function
        | import llvm
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %a : number.Float64 = 1.0
        |     %b : number.Float64 = 2.0
        |     %c : Nil = llvm.fadd(%a, %b)
    """)
    fadd = next(op for op in block.ops if isinstance(op, llvm.FaddOp))
    deps = list(runtime_dependencies(fadd))
    # fadd depends on %a and %b — both should appear
    dep_names = [v.name for v in deps if v.name is not None]
    assert "a" in dep_names
    assert "b" in dep_names


def test_runtime_dependencies_excludes_type_deps():
    """runtime_dependencies does NOT follow type edges (only operands + captures)."""
    block = _parse("""
        | import function
        | import llvm
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %a : number.Float64 = 1.0
        |     %b : number.Float64 = 2.0
        |     %c : Nil = llvm.fadd(%a, %b)
    """)
    fadd = next(op for op in block.ops if isinstance(op, llvm.FaddOp))
    deps = list(runtime_dependencies(fadd))
    # Type values (Float64, Nil) should NOT appear in runtime deps
    from dgen import Type

    type_deps = [v for v in deps if isinstance(v, Type)]
    assert type_deps == []


def test_runtime_dependencies_follows_captures():
    """runtime_dependencies follows block captures as runtime edges."""
    block = _parse("""
        | import function
        | import goto
        | import index
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %x : index.Index = 42
        |     %lbl : goto.Label = goto.label([]) body() captures(%x):
        |         %_ : Nil = ()
    """)
    label = next(op for op in block.ops if isinstance(op, goto.LabelOp))
    deps = list(runtime_dependencies(label))
    dep_names = [v.name for v in deps if v.name is not None]
    assert "x" in dep_names


def test_runtime_dependencies_no_duplicates():
    """Each value appears at most once even when referenced by multiple ops."""
    block = _parse("""
        | import function
        | import llvm
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %a : number.Float64 = 1.0
        |     %b : Nil = llvm.fadd(%a, %a)
    """)
    fadd = next(op for op in block.ops if isinstance(op, llvm.FaddOp))
    deps = list(runtime_dependencies(fadd))
    assert len(deps) == len(set(deps))


def test_runtime_dependencies_empty_for_constant():
    """A constant has no runtime dependencies."""
    block = _parse("""
        | import function
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %a : number.Float64 = 42.0
    """)
    from dgen import Constant

    const = next(op for op in block.ops if isinstance(op, Constant))
    deps = list(runtime_dependencies(const))
    assert deps == []


def test_runtime_dependencies_transitive():
    """Dependencies are transitive: a -> b -> c yields [a, b]."""
    block = _parse("""
        | import function
        | import llvm
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %a : number.Float64 = 1.0
        |     %b : number.Float64 = 2.0
        |     %c : Nil = llvm.fadd(%a, %b)
        |     %d : Nil = llvm.fmul(%c, %a)
    """)
    fmul = next(op for op in block.ops if isinstance(op, llvm.FmulOp))
    deps = list(runtime_dependencies(fmul))
    dep_names = [v.name for v in deps if v.name is not None]
    # %d depends on %c which depends on %a, %b — all three should appear
    assert "a" in dep_names
    assert "b" in dep_names
    assert "c" in dep_names


def test_runtime_dependencies_topological_order():
    """Dependencies are yielded before the values that depend on them."""
    block = _parse("""
        | import function
        | import llvm
        | import number
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %a : number.Float64 = 1.0
        |     %b : number.Float64 = 2.0
        |     %c : Nil = llvm.fadd(%a, %b)
        |     %d : Nil = llvm.fmul(%c, %a)
    """)
    fmul = next(op for op in block.ops if isinstance(op, llvm.FmulOp))
    deps = list(runtime_dependencies(fmul))
    idx = {v: i for i, v in enumerate(deps)}
    # %a and %b must appear before %c (fadd uses them)
    fadd = next(v for v in deps if v.name == "c")
    a = next(v for v in deps if v.name == "a")
    b = next(v for v in deps if v.name == "b")
    assert idx[a] < idx[fadd]
    assert idx[b] < idx[fadd]
    # %c must appear before %d (fmul uses it) — %d itself is not in deps
    # since runtime_dependencies yields deps OF the value, not the value itself


# ---------------------------------------------------------------------------
# emit_label (structural — the function has known WIP syntax issues,
# so these test the shape of what it should produce)
# ---------------------------------------------------------------------------


def test_emit_label_registered_for_label_op():
    """The emitter for goto.LabelOp is the emit_label_op function."""
    from dgen.llvm.codegen import emit_label_op

    assert EMITTERS[goto.LabelOp] is emit_label_op


def test_emit_dispatches_by_value_class():
    """emit() dispatches to the emitter registered for type(value)."""
    from dgen.llvm.codegen import emit

    @dataclass(eq=False, kw_only=True)
    class _Sentinel(dgen.Op):
        type: dgen.Value = field(default_factory=builtin.Nil)

    @emitter_for(_Sentinel)
    def handle_sentinel(value):
        yield "sentinel_output"

    dummy = _Sentinel()
    lines = list(emit(dummy))
    # emit() prepends %name = for value-producing ops
    assert len(lines) == 1
    assert "sentinel_output" in lines[0]

    del EMITTERS[_Sentinel]


def test_emit_linearized_nested_loop():
    """emit_linearized handles nested loop IR (lowered to LLVM ops) without crashing."""
    value = asm.parse(
        strip_prefix("""
        | import function
        | import goto
        | import index
        | import llvm
        | import number
        |
        | %test : function.Function<[], Nil> = function.function<Nil>() body():
        |     %0 : index.Index = 0
        |     %loop_header0 : goto.Label = goto.region([%0]) body<%self: goto.Label, %exit0: goto.Label>(%i0: index.Index):
        |         %1 : index.Index = 2
        |         %2 : number.Boolean = llvm.icmp<String("slt")>(%i0, %1)
        |         %loop_body0 : goto.Label = goto.label([]) body(%j0: index.Index) captures(%self):
        |             %3 : index.Index = 1
        |             %4 : index.Index = llvm.add(%j0, %3)
        |             %5 : index.Index = 0
        |             %loop_header1 : goto.Label = goto.region([%5]) body<%6: goto.Label, %exit1: goto.Label>(%i1: index.Index):
        |                 %7 : index.Index = 2
        |                 %8 : number.Boolean = llvm.icmp<String("slt")>(%i1, %7)
        |                 %loop_body1 : goto.Label = goto.label([]) body(%j1: index.Index) captures(%6):
        |                     %9 : index.Index = 1
        |                     %10 : index.Index = llvm.add(%j1, %9)
        |                     %11 : index.Index = 0
        |                     %12 : Nil = chain(%11, %11)
        |                     %13 : index.Index = chain(%10, %12)
        |                     %14 : Nil = goto.branch<%6>([%13])
        |                 %15 : Nil = goto.conditional_branch<%loop_body1, %exit1>(%8, [%i1], [])
        |             %16 : index.Index = chain(%4, %loop_header1)
        |             %17 : Nil = goto.branch<%self>([%16])
        |         %18 : Nil = goto.conditional_branch<%loop_body0, %exit0>(%2, [%i0], [])
        """)
    )

    emitted = list(emit(value))
    # Should produce some output lines (labels, instructions)
    assert len(emitted) > 0


# ---------------------------------------------------------------------------
# Extern discovery
# ---------------------------------------------------------------------------


def test_externs_function_with_typed_args():
    """_externs discovers a function extern with real argument/return types (malloc)."""
    value = asm.parse(
        strip_prefix("""
        | import function
        | import index
        | import algebra
        | import llvm
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %malloc : function.Function<[index.Index], llvm.Ptr> = extern<String("malloc")>()
        |     %size : index.Index = 48
        |     %ptr : llvm.Ptr = function.call(%malloc, [%size])
    """)
    )
    externs = _externs(value)
    assert len(externs) == 1


def test_externs_non_function():
    """_externs discovers a non-function extern (global value)."""
    value = asm.parse(
        strip_prefix("""
        | import function
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %greeting : String = extern<String("hello_world")>()
    """)
    )
    externs = _externs(value)
    assert len(externs) == 1


def test_externs_nested_in_region():
    """_externs finds externs nested inside a region body."""
    value = asm.parse(
        strip_prefix("""
        | import function
        | import goto
        | import index
        | import llvm
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %init : index.Index = 0
        |     %r : goto.Label = goto.region([%init]) body<%self: goto.Label, %exit: goto.Label>(%i: index.Index):
        |         %print : function.Function<[llvm.Ptr, index.Index], Nil> = extern<String("print_memref")>()
        |         %0 : Nil = function.call(%print, [])
        |         %1 : index.Index = 1
        |         %next : index.Index = llvm.add(%i, %1)
        |         %next2 : index.Index = chain(%next, %0)
        |         %_ : Nil = goto.branch<%self>([%next2])
    """)
    )
    externs = _externs(value)
    assert len(externs) == 1


def test_externs_no_duplicates():
    """_externs deduplicates: same ExternOp referenced twice appears once."""
    value = asm.parse(
        strip_prefix("""
        | import function
        | import index
        | import algebra
        | import llvm
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %malloc : function.Function<[index.Index], llvm.Ptr> = extern<String("malloc")>()
        |     %0 : llvm.Ptr = function.call(%malloc, [])
        |     %1 : llvm.Ptr = function.call(%malloc, [])
        |     %_ : Nil = chain(%0, %1)
    """)
    )
    externs = _externs(value)
    assert len(externs) == 1


def test_externs_dedup_distinct_instances_same_symbol():
    """_externs deduplicates by symbol name, not object identity.

    Lowering passes create a fresh ExternOp per call site. Two different
    ExternOp instances for "malloc" must produce a single declare.
    """
    from dgen.dialects.builtin import ExternOp, String
    from dgen.dialects.function import Function

    malloc1 = ExternOp(
        symbol=String().constant("malloc"),
        type=Function(arguments=pack([index.Index()]), result_type=llvm.Ptr()),
    )
    malloc2 = ExternOp(
        symbol=String().constant("malloc"),
        type=Function(arguments=pack([index.Index()]), result_type=llvm.Ptr()),
    )
    assert malloc1 is not malloc2

    from dgen.block import BlockArgument
    from dgen.dialects.function import FunctionOp

    arg = BlockArgument(name="n", type=index.Index())
    call1 = function.CallOp(callee=malloc1, arguments=pack([arg]), type=llvm.Ptr())
    call2 = function.CallOp(callee=malloc2, arguments=pack([arg]), type=llvm.Ptr())
    result = builtin.ChainOp(lhs=call1, rhs=call2, type=llvm.Ptr())
    func = FunctionOp(
        name="f",
        body=dgen.Block(result=result, args=[arg]),
        result_type=llvm.Ptr(),
        type=Function(arguments=pack([index.Index()]), result_type=llvm.Ptr()),
    )
    value = func

    externs = _externs(value)
    assert len(externs) == 1


def test_externs_multiple_distinct():
    """_externs finds multiple distinct externs."""
    value = asm.parse(
        strip_prefix("""
        | import function
        | import index
        | import algebra
        | import llvm
        |
        | %f : function.Function<[], Nil> = function.function<Nil>() body():
        |     %malloc : function.Function<[index.Index], llvm.Ptr> = extern<String("malloc")>()
        |     %print : function.Function<[llvm.Ptr, index.Index], Nil> = extern<String("print_memref")>()
        |     %ptr : llvm.Ptr = function.call(%malloc, [])
        |     %0 : Nil = function.call(%print, [])
        |     %_ : Nil = chain(%ptr, %0)
    """)
    )
    externs = _externs(value)
    assert len(externs) == 2


# ---------------------------------------------------------------------------
# jit_function + call
# ---------------------------------------------------------------------------


def test_call_invokes_jit_function():
    """jit_function returns a ConstantOp[Function]; call invokes it."""
    from dgen.llvm.codegen import call
    from dgen.testing import llvm_compile as codegen_compile

    value = asm.parse(
        strip_prefix("""
        | import function
        | import index
        | import algebra
        |
        | %add : function.Function<[index.Index, index.Index], index.Index> = function.function<index.Index>() body(%a: index.Index, %b: index.Index):
        |     %r : index.Index = algebra.add(%a, %b)
    """)
    )
    exe = codegen_compile(value)
    # Classic path.
    assert exe.run(3, 4).to_json() == 7
    # New path: the func_constant behaves like a ConstantOp[Function].
    fc = exe.func_constant
    assert isinstance(fc.type, function.Function)
    assert call(fc, 5, 6).to_json() == 11


def test_call_function_constant_keeps_engine_alive():
    """The returned ConstantOp[Function] can be called after the Executable is gone."""
    from dgen.llvm.codegen import call
    from dgen.testing import llvm_compile as codegen_compile

    value = asm.parse(
        strip_prefix("""
        | import function
        | import index
        | import algebra
        |
        | %double : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.add(%x, %x)
    """)
    )
    fc = codegen_compile(value).func_constant
    # Executable is now unreachable; engine lives in fc.value.origins.
    assert call(fc, 21).to_json() == 42


def test_llvm_codegen_on_extern():
    """``LLVMCodegen().run`` on an ExternOp builds a trampoline that invokes it.

    An ``ExternOp`` has ``Function`` type but no body, so it can't be used
    directly as the entry. The codegen wraps it in a nil-body trampoline
    that forwards args to the extern symbol — making the resulting
    Executable callable with the extern's signature.
    """
    from dgen.llvm.codegen import LLVMCodegen

    value = asm.parse(
        strip_prefix("""
        | import function
        | import index
        | import llvm
        | %m : function.Function<[index.Index], llvm.Ptr> = extern<String("malloc")>()
    """)
    )
    exe = LLVMCodegen().run(value)
    assert "declare ptr @malloc(i64)" in exe.ir
    assert "define ptr @main(i64" in exe.ir
    assert exe.main_name == "main"
    # malloc actually runs and returns a non-zero pointer.
    result = exe.run(16)
    (raw,) = result.unpack()
    assert isinstance(raw, int) and raw != 0
