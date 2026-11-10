"""Compiler: staging-aware pass pipeline with exit pass."""

from __future__ import annotations

from collections.abc import Iterator
from contextvars import ContextVar
from typing import Generic, Protocol, TypeVar

import dgen
from dgen.passes.pass_ import LoweringError, Ops, Pass, name, ops, value_classes

verify_passes: ContextVar[bool] = ContextVar("dgen.verify_passes", default=False)

T = TypeVar("T", covariant=True)


class PipelineError(Exception):
    """A pipeline leaves ops that no later pass lowers and the exit doesn't accept."""


class ExitPass(Protocol, Generic[T]):
    """Terminal pass that converts Value → T (e.g. Executable)."""

    def run(self, value: dgen.Value) -> T: ...

    def accepts(self, op: type[dgen.Op]) -> bool: ...


class IdentityPass:
    def run(self, value: dgen.Value) -> dgen.Value:
        return value

    def accepts(self, op: type[dgen.Op]) -> bool:
        return True


class Compiler(Generic[T]):
    """Staging-aware compilation pipeline.

    Resolves comptime boundaries (staging), runs passes, and
    applies an exit pass to produce the final result.

    When all comptime boundaries are stage-0 (evaluable without runtime
    values), staging resolves them and the pipeline proceeds normally.
    When stage-1+ boundaries remain (runtime-dependent), the compiler
    builds callback thunks that JIT-compile at runtime.

    Invariants:
      - every op a pass emits is eliminated by a later pass or accepted by exit
      - so is every op in inputs, when the pipeline names them
    """

    def __init__(
        self, passes: list[Pass], exit: ExitPass[T], *, inputs: Ops | None = None
    ) -> None:
        self.passes = passes
        self.exit = exit
        self.inputs = inputs
        if unlowered := list(self._unlowered()):
            raise PipelineError("\n".join(unlowered))

    def _unlowered(self) -> Iterator[str]:
        """A line for each op some step emits that nothing after it takes."""
        steps = [
            (type(p).__name__, p.emitted, self.passes[i + 1 :])
            for i, p in enumerate(self.passes)
        ]
        if self.inputs is not None:
            steps.insert(0, ("the frontend", ops(self.inputs), self.passes))
        for step, emitted, later in steps:
            lowered = set().union(*(p.eliminated for p in later))
            for op in sorted(emitted - lowered, key=name):
                if not self.exit.accepts(op):
                    yield (
                        f"{step} emits {name(op)}, which no later pass lowers "
                        f"and {type(self.exit).__name__} does not accept"
                    )

    def compile(self, value: dgen.Value) -> T:
        """Full pipeline: staging → passes → exit."""
        from dgen.staging import compile_value

        return compile_value(value, self)

    def run(self, value: dgen.Value) -> T:
        """Run passes + exit (no staging)."""
        verify = verify_passes.get()
        if verify and self.inputs is not None:
            classes = {c for c in value_classes(value) if issubclass(c, dgen.Op)}
            if unexpected := classes - ops(self.inputs):
                raise LoweringError(
                    f"the input holds {', '.join(sorted(map(name, unexpected)))}, "
                    "which the pipeline's inputs don't name"
                )
        for i, p in enumerate(self.passes):
            before = value_classes(value) if verify else set()
            if verify:
                p.verify_preconditions(value)
            continuation = Compiler(self.passes[i + 1 :], self.exit)
            value = p.run(value, continuation)
            if verify:
                p.verify_declarations(before, value)
                p.verify_postconditions(value)
        return self.exit.run(value)
