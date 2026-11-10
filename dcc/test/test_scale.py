"""Pass time on one large C function, for measuring replacement cost."""

import time

import llvmlite.binding as llvm_binding
import pytest

from dcc.parser.c_parser import parse_c_string
from dcc.parser.lowering import lower
from dcc.passes.c_to_llvm import CToLLVM
from dcc.passes.c_to_memory import CToMemory
from dgen.codegen import emit_llvm_ir
from dgen.compiler import Compiler, IdentityPass, verify_passes
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.memory_to_llvm import MemoryToLLVM
from dgen.passes.pass_ import Pass

PASSES: list[Pass] = [
    CToMemory(),
    CToLLVM(),
    AlgebraToLLVM(),
    MemoryToLLVM(),
    ControlFlowToGoto(),
]


def _straight_line_c(statements: int) -> str:
    """One function of `statements` updates to four locals, each a load and a store."""
    body = "\n".join(f"  x{i % 4} = x{(i + 1) % 4} + {i};" for i in range(statements))
    return f"int f(int a) {{\n  int x0 = a, x1 = 1, x2 = 2, x3 = 3;\n{body}\n  return x0;\n}}\n"


@pytest.mark.slow
@pytest.mark.parametrize("statements", [250, 500, 1000])
def test_large_function_pass_time(statements: int):
    _, stats = lower(parse_c_string(_straight_line_c(statements)))
    (value,) = stats.function_ops
    times: list[str] = []
    token = verify_passes.set(False)
    try:
        for p in PASSES:
            start = time.perf_counter()
            value = Compiler([p], IdentityPass()).run(value)
            times.append(f"{type(p).__name__} {time.perf_counter() - start:.2f}s")
    finally:
        verify_passes.reset(token)
    print(f"\n{statements} statements: {', '.join(times)}")
    llvm_binding.initialize_native_target()
    llvm_binding.initialize_native_asmprinter()
    ir, _ = emit_llvm_ir(value)
    llvm_binding.parse_assembly(ir).verify()
