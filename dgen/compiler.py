"""Compiler: staging-aware pass pipeline with exit pass."""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
from typing import Generic, Protocol, TypeVar

import dgen
from dgen.block import BlockArgument
from dgen.module import ConstantOp, Module, pack
from dgen.passes.pass_ import Pass
from dgen.type import Memory

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

    def compile_value(
        self,
        value: dgen.Value,
        *,
        block_args: Sequence[BlockArgument] = (),
        args: Sequence[object] = (),
    ) -> ConstantOp:
        """Compile a value to a constant: wrap in function, JIT, execute.

        Wraps the target value in a mini function+module, runs the full
        pass pipeline + exit to get an Executable, executes it, and returns
        the result as a ConstantOp.
        """
        from dgen.codegen import Executable
        from dgen.dialects.function import Function, FunctionOp

        func = FunctionOp(
            name="main",
            body=dgen.Block(result=value, args=list(block_args)),
            result_type=value.type,
            type=Function(
                arguments=pack(arg.type for arg in block_args),
                result_type=value.type,
            ),
        )
        module = Module(ops=[func])
        exe = self.run(module)
        assert isinstance(exe, Executable)
        # LIFETIME BUG WORKAROUND: Create Memory objects before the call so
        # they outlive exe.run(). See TODO.md for the proper fix.
        memories = [
            Memory.from_value(param.type, arg) for arg, param in zip(args, block_args)
        ]
        result = exe.run(*memories)
        json_value = result.to_json()
        const_type = value.type
        if isinstance(json_value, dict) and "tag" in json_value:
            const_type = dgen.type.TypeType()
        return ConstantOp(value=json_value, type=const_type)

    def run(self, module: Module) -> T:
        """Run passes + exit (no staging)."""
        for i, p in enumerate(self.passes):
            if verify_passes.get():
                p.verify_preconditions(module)
            continuation = Compiler(self.passes[i + 1 :], self.exit)
            module = p.run(module, continuation)
            if verify_passes.get():
                p.verify_postconditions(module)
        return self.exit.run(module)
