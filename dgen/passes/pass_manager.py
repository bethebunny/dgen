"""PassManager: sequential execution of passes with optional verification."""

from __future__ import annotations

from dgen.module import Module
from dgen.passes.pass_ import Pass


class PassManager:
    def __init__(self, passes: list[Pass], *, verify: bool = False) -> None:
        self._passes = passes
        self._verify = verify

    def run(self, module: Module) -> Module:
        for p in self._passes:
            if self._verify:
                p.verify_preconditions(module)
            module = p.run(module)
            if self._verify:
                p.verify_postconditions(module)
        return module
