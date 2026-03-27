"""Lower builtin dialect ops to LLVM dialect ops.

Handles: CallOp → llvm.call.
"""

from __future__ import annotations

from dgen.dialects import function, llvm
from dgen.dialects.builtin import String
from dgen.module import PackOp
from dgen.passes.pass_ import Pass, Rewriter, lowering_for


class BuiltinToLLVMLowering(Pass):
    allow_unregistered_ops = True

    @lowering_for(function.CallOp)
    def lower_call(self, op: function.CallOp, rewriter: Rewriter) -> bool:
        callee_name = op.callee.name
        assert callee_name is not None
        if isinstance(op.arguments, PackOp):
            args = list(op.arguments.values)
        else:
            args = [op.arguments]
        pack = PackOp(values=args, type=op.arguments.type)
        llvm_call = llvm.CallOp(
            callee=String().constant(callee_name),
            args=pack,
            type=op.type,
        )
        rewriter.replace_uses(op, llvm_call)
        return True
