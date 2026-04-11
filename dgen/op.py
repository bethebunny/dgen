from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from .dialect import Dialect
from .type import Value


@dataclass(eq=False)
class Op(Value):
    """Base class for all dialect operations."""

    # `name` is kw_only so subclasses can declare positional fields without
    # colliding with this default — otherwise every subclass dataclass would
    # need kw_only=True to avoid "non-default argument follows default".
    name: str | None = field(default=None, kw_only=True)
    asm_name: ClassVar[str]
    dialect: ClassVar[Dialect]
