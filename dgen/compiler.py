"""Compiler: staging-aware pass pipeline with exit pass."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Generic, Protocol, TypeVar

import dgen
from dgen.dialects.function import Function, FunctionOp
from dgen.module import ConstantOp, Module, pack
from dgen.passes.pass_ import Pass
from dgen.type import Constant

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

    def compile(self, value: dgen.Value) -> Constant:
        """Evaluate ``value`` and return it as a ``Constant``.

        Wraps ``value`` in a zero-arg function that returns it, runs the
        compiler's passes + exit pass to produce an ``Executable``, invokes
        the executable, and wraps the resulting ``Memory`` as a ``ConstantOp``.
        """
        wrapper = FunctionOp(
            name="main",
            body=dgen.Block(result=value),
            result_type=value.type,
            type=Function(arguments=pack([]), result_type=value.type),
        )
        exe = self.run(Module(ops=[wrapper]))
        mem = exe.run()  # type: ignore[attr-defined]
        return ConstantOp(type=value.type, value=mem)

    def compile_module(self, module: Module) -> T:
        """Full pipeline on a Module: staging → passes → exit."""
        from dgen.staging import compile_module

        return compile_module(module, self)

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

    def run_value(self, value: dgen.Value) -> dgen.Value:
        """Run passes on a single root value (no module, no exit pass).

        Each pass runs on ``value`` with a continuation whose passes are the
        remaining ones, and the output of each pass feeds the next.
        """
        for i, p in enumerate(self.passes):
            continuation = Compiler(self.passes[i + 1 :], self.exit)
            value = p.run(value, continuation)
        return value
