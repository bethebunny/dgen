from pathlib import Path

from .block import Block
from .dialect import Dialect
from .op import Op
from .trait import Trait
from .type import Constant, Type, TypeType, Value

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

from dgen.spec.importer import install as _install_dgen_hook

_install_dgen_hook()

# Register the core dialect directory so ``Dialect.get("llvm")`` etc. work.
Dialect.paths.append(Path(__file__).parent / "dialects")
