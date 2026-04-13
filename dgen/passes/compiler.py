"""Compiler: staging-aware pass pipeline with exit pass."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Generic, Protocol, TypeVar

import dgen
from dgen.passes.pass_ import Pass

verify_passes: ContextVar[bool] = ContextVar(
    "dgen.ir.verification_passes", default=False
)

T = TypeVar("T", covariant=True)


class ExitPass(Protocol, Generic[T]):
    """Terminal pass that converts Value → T (e.g. Executable)."""

    def run(self, value: dgen.Value) -> T: ...


class IdentityPass:
    def run(self, value: dgen.Value) -> dgen.Value:
        return value


class Compiler(Generic[T]):
    """Staging-aware compilation pipeline.

    Resolves comptime boundaries (staging), runs passes, and
    applies an exit pass to produce the final result.

    When all comptime boundaries are stage-0 (evaluable without runtime
    values), staging resolves them and the pipeline proceeds normally.
    When stage-1+ boundaries remain (runtime-dependent), the compiler
    builds callback thunks that JIT-compile at runtime.

    A ``Compiler[Value]`` (with ``IdentityPass`` as exit) satisfies the
    ``Pass`` interface, so compilers can be nested as passes inside other
    compilers. This enables grouping passes into reusable sub-pipelines.
    """

    def __init__(self, passes: list[Pass], exit: ExitPass[T]) -> None:
        self.passes = passes
        self.exit = exit

    def compile(self, value: dgen.Value) -> T:
        """Full pipeline: staging → passes → exit."""
        from dgen.passes.staging import compile_value

        return compile_value(value, self)

    def run(self, value: dgen.Value, compiler: Compiler[object] | None = None) -> T:
        """Run passes + exit (no staging).

        The optional *compiler* argument makes ``Compiler[Value]``
        usable as a ``Pass`` inside another compiler's pass list.
        When used as a pass, the outer compiler's continuation is
        ignored — this compiler runs its own complete pipeline.
        """
        for i, p in enumerate(self.passes):
            if verify_passes.get():
                p.verify_preconditions(value)
            continuation = Compiler(self.passes[i + 1 :], self.exit)
            value = p.run(value, continuation)
            if verify_passes.get():
                p.verify_postconditions(value)
        return self.exit.run(value)

    def verify_preconditions(self, value: dgen.Value) -> None:
        """Delegate to the first pass, if any."""
        if self.passes:
            self.passes[0].verify_preconditions(value)

    def verify_postconditions(self, value: dgen.Value) -> None:
        """Delegate to the last pass, if any."""
        if self.passes:
            self.passes[-1].verify_postconditions(value)
