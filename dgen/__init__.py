from pathlib import Path

from .block import Block
from .dialect import Dialect
from .op import Op
from .type import Constant, Trait, Type, TypeType, Value

__all__ = [
    "Block",
    "Constant",
    "Dialect",
    "Op",
    "Trait",
    "Type",
    "TypeType",
    "Value",
]

from dgen.gen.importer import install as _install_dgen_hook

_install_dgen_hook()

# Register the core dialect directory so ``Dialect.get("llvm")`` etc. work.
Dialect.paths.append(Path(__file__).parent / "dialects")
