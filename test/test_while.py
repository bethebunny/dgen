"""Tests for control_flow.while: ASM round-trip, lowering to goto, and codegen.

WhileOp semantics:
- initial_arguments: pack of initial values for loop-carried variables
- condition block: receives loop vars as block args, result is Index (0=false, nonzero=true)
- body block: receives loop vars as block args, result is pack of next-iteration values
- WhileOp result: Nil
"""

from dgen import asm
from dgen.asm.parser import parse
from dgen.llvm.codegen import LLVMCodegen
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.dialects import control_flow
from dgen.llvm.algebra_to_llvm import AlgebraToLLVM
from dgen.llvm.builtin_to_llvm import BuiltinToLLVM
from dgen.llvm.memory_to_llvm import MemoryToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.record_to_memory import RecordToMemory
from dgen.testing import assert_ir_equivalent, strip_prefix


# -- ASM round-trip tests --


def test_while_basic_roundtrip():
    """Simple while loop: count from 0 to 10."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import control_flow
        | import index
        | %zero : index.Index = 0
        | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |     %ten : index.Index = 10
        |     %cmp : number.Boolean = algebra.less_than(%i, %ten)
        | body(%i: index.Index):
        |     %one : index.Index = 1
        |     %next : index.Index = algebra.add(%i, %one)
    """)
    while_op = parse(ir)
    assert isinstance(while_op, control_flow.WhileOp)
    assert len(while_op.condition.args) == 1
    assert while_op.condition.args[0].name == "i"
    assert len(while_op.body.args) == 1
    assert while_op.body.args[0].name == "i"
    assert_ir_equivalent(while_op, asm.parse(asm.format(while_op)))


def test_while_with_capture_roundtrip():
    """While loop that captures a value from the enclosing scope."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import control_flow
        | import index
        | import function
        | %main : function.Function<[index.Index], Nil> = function.function<Nil>() body(%limit: index.Index):
        |     %zero : index.Index = 0
        |     %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
        |         %cmp : number.Boolean = algebra.less_than(%i, %limit)
        |     body(%i: index.Index):
        |         %one : index.Index = 1
        |         %next : index.Index = algebra.add(%i, %one)
    """)
    value = parse(ir)
    while_op = value.body.result
    assert isinstance(while_op, control_flow.WhileOp)
    assert len(while_op.condition.args) == 1
    assert len(while_op.body.args) == 1
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_while_no_ivs_roundtrip():
    """While loop with no loop-carried variables (infinite loop style)."""
    ir = strip_prefix("""
        | import control_flow
        | import index
        | %loop : Nil = control_flow.while([]) condition():
        |     %cond : index.Index = 1
        | body():
        |     %nop : index.Index = 0
    """)
    while_op = parse(ir)
    assert isinstance(while_op, control_flow.WhileOp)
    assert while_op.condition.args == []
    assert while_op.body.args == []
    assert_ir_equivalent(while_op, asm.parse(asm.format(while_op)))


# -- Lowering tests (while → goto) --


SIMPLE_WHILE = strip_prefix("""
    | import algebra
    | import control_flow
    | import index
    | import number
    | import record
    | %zero : index.Index = 0
    | %loop : Nil = control_flow.while([%zero]) condition(%i: index.Index):
    |     %ten : index.Index = 10
    |     %cmp : number.Boolean = algebra.less_than(%i, %ten)
    | body(%i: index.Index):
    |     %one : index.Index = 1
    |     %next : index.Index = algebra.add(%i, %one)
    |     %carry : Tuple<[index.Index]> = record.pack([%next])
""")


def test_while_lowering_to_goto(ir_snapshot):
    """WhileOp lowered to goto labels produces expected IR."""
    m = parse(SIMPLE_WHILE)
    lowered = Compiler([ControlFlowToGoto()], IdentityPass()).compile(m)
    assert lowered == ir_snapshot


def test_while_llvm_ir(snapshot):
    """WhileOp all the way to LLVM IR."""
    m = parse(SIMPLE_WHILE)
    exe = Compiler(
        [
            ControlFlowToGoto(),
            RecordToMemory(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        LLVMCodegen(),
    ).compile(m)
    assert exe.ir == snapshot


NESTED_WHILE = strip_prefix("""
    | import algebra
    | import control_flow
    | import index
    | import number
    | import record
    | %zero : index.Index = 0
    | %outer : Nil = control_flow.while([%zero]) condition(%i: index.Index):
    |     %two : index.Index = 2
    |     %cmp : number.Boolean = algebra.less_than(%i, %two)
    | body(%oi: index.Index):
    |     %izero : index.Index = 0
    |     %inner : Nil = control_flow.while([%izero]) condition(%j: index.Index):
    |         %jtwo : index.Index = 2
    |         %jcmp : number.Boolean = algebra.less_than(%j, %jtwo)
    |     body(%j2: index.Index):
    |         %nop : index.Index = 0
    |         %nop2 : Nil = chain(%nop, %nop)
    |         %icarry : Tuple<[index.Index]> = record.pack([%nop])
    |     %one : index.Index = 1
    |     %next : index.Index = algebra.add(%oi, %one)
    |     %next2 : index.Index = chain(%next, %inner)
    |     %ocarry : Tuple<[index.Index]> = record.pack([%next2])
""")


def test_nested_while_lowering(ir_snapshot):
    """Nested WhileOps lowered to goto labels."""
    m = parse(NESTED_WHILE)
    lowered = Compiler([ControlFlowToGoto()], IdentityPass()).compile(m)
    assert lowered == ir_snapshot


def test_nested_while_llvm_ir(snapshot):
    """Nested while loops all the way to LLVM IR."""
    m = parse(NESTED_WHILE)
    exe = Compiler(
        [
            ControlFlowToGoto(),
            RecordToMemory(),
            MemoryToLLVM(),
            BuiltinToLLVM(),
            AlgebraToLLVM(),
        ],
        LLVMCodegen(),
    ).compile(m)
    assert exe.ir == snapshot
