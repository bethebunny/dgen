"""record dialect lowering — currently a no-op.

After the State-effect / Linear Reference refactor, ``record.pack`` and
``record.get`` survive to LLVM codegen as runtime aggregates: codegen
emits an ``insertvalue`` chain for ``record.pack`` and an
``extractvalue`` for ``record.get``. No memory allocation is needed for
records — they're SSA aggregates throughout the pipeline.

The pass remains in the pipeline as a placeholder (and to keep its name
stable for users who explicitly compose passes); it currently performs
no transformations.
"""

from __future__ import annotations

from dgen.passes.pass_ import Pass


class RecordToMemory(Pass):
    allow_unregistered_ops = True
