"""Compiler: staging-aware pass pipeline with exit pass."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Generic, Protocol, TypeVar

from dgen.module import Module
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

    def run(self, module: Module, *, verify: bool = False) -> Module:
        """Run passes only (no staging, no exit pass)."""
        token = verify_passes.set(True) if verify and not verify_passes.get() else None
        try:
            for i, p in enumerate(self.passes):
                if verify_passes.get():
                    p.verify_preconditions(module)
                continuation = Compiler(self.passes[i + 1 :], self.exit)
                module = p.run(module, continuation)
                if verify_passes.get():
                    p.verify_postconditions(module)
        finally:
            if token is not None:
                verify_passes.reset(token)
        return module
