"""LLVM codegen emitters for the C dialect.

Registers an emitter for ``c.CReturnOp`` — used when a ``return``
statement appears inside a control-flow body and the frontend can't
funnel the value through the function's natural block result.

``CReturnOp`` is Never-typed, so ``emit_linearized`` treats it as a
basic-block terminator and ``emit`` skips the ``%name =`` SSA prefix.
"""

from __future__ import annotations

from collections.abc import Iterator

from dgen.llvm.codegen import emitter_for, llvm_type, value_reference

from dcc.dialects.c import CReturnOp


@emitter_for(CReturnOp)
def emit_c_return(op: CReturnOp) -> Iterator[str]:
    ret_type = llvm_type(op.value.type)
    val = value_reference(op.value)
    yield f"  ret {ret_type} {val}"
