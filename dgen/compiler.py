"""Compiler: staging-aware pass pipeline with exit pass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from dgen.module import Module
from dgen.passes.pass_ import Pass

T = TypeVar("T")


class ExitPass(ABC, Generic[T]):
    """Terminal pass that converts Module → T (e.g. Executable)."""

    @abstractmethod
    def run(self, module: Module) -> T: ...


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
        for p in self.passes:
            if verify:
                p.verify_preconditions(module)
            module = p.run(module)
            if verify:
                p.verify_postconditions(module)
        return module
