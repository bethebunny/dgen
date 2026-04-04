"""Compiler: staging-aware pass pipeline with exit pass."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Generic, Protocol, TypeVar

import dgen
from dgen.dialects.function import Function, FunctionOp
from dgen.module import ConstantOp, Module, pack
from dgen.passes.pass_ import Pass

verify_passes: ContextVar[bool] = ContextVar("dgen.verify_passes", default=False)

T = TypeVar("T", covariant=True)


class ExitPass(Protocol, Generic[T]):
    """Terminal pass that converts Module → T (e.g. Executable)."""

    def run(self, module: Module) -> T: ...


class IdentityPass:
    def run(self, module: Module) -> Module:
        return module


class Compiler(Generic[T]):
    """Staging-aware compilation pipeline.

    Resolves comptime boundaries (staging), runs passes, and
    applies an exit pass to produce the final result.

    When all comptime boundaries are stage-0 (evaluable without runtime
    values), staging resolves them and the pipeline proceeds normally.
    When stage-1+ boundaries remain (runtime-dependent), the compiler
    builds callback thunks that JIT-compile at runtime.
    """

    def __init__(self, passes: list[Pass], exit: ExitPass[T]) -> None:
        self.passes = passes
        self.exit = exit

    def compile(self, module: Module) -> T:
        """Full pipeline: staging → passes → exit."""
        from dgen.staging import compile_module

        return compile_module(module, self)

    def compile_value(self, value: dgen.Value) -> ConstantOp:
        """Compile a stage-0 value to a constant: wrap in function, JIT, execute.

        The value must have no runtime (BlockArgument) dependencies. Callers
        with runtime values should substitute them as ConstantOps in the IR
        before calling this.
        """
        func = FunctionOp(
            name="main",
            body=dgen.Block(result=value),
            result_type=value.type,
            type=Function(arguments=pack([]), result_type=value.type),
        )
        exe = self.run(Module(ops=[func]))
        result = exe.run()  # type: ignore[attr-defined]
        return ConstantOp(value=result.to_json(), type=value.type)

    def run(self, module: Module) -> T:
        """Run passes + exit (no staging)."""
        for i, p in enumerate(self.passes):
            if verify_passes.get():
                p.verify_preconditions(module)
            continuation = Compiler(self.passes[i + 1 :], self.exit)
            module = Module(ops=[p.run(op, continuation) for op in module.ops])
            if verify_passes.get():
                p.verify_postconditions(module)
        return self.exit.run(module)
