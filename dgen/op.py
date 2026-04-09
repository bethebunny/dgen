from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .dialect import Dialect
from .type import Value


@dataclass(eq=False)
class Op(Value):
    """Base class for all dialect operations."""

    name: str | None = None
    asm_name: ClassVar[str]
    dialect: ClassVar[Dialect]
