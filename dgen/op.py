from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from .dialect import Dialect
from .type import Value

if TYPE_CHECKING:
    from .ir.verification import BlockLinearityContext


@dataclass(eq=False)
class Op(Value):
    """Base class for all dialect operations."""

    # `name` is kw_only so subclasses can declare positional fields without
    # colliding with this default — otherwise every subclass dataclass would
    # need kw_only=True to avoid "non-default argument follows default".
    name: str | None = field(default=None, kw_only=True)
    asm_name: ClassVar[str]
    dialect: ClassVar[Dialect]

    def required_dialects(self) -> Iterator[Dialect]:
        yield self.dialect
        yield from super().required_dialects()

    def verify_block_linearity(self, ctx: BlockLinearityContext) -> None:
        """Charge this op's child-block captures in the enclosing
        block's linearity context Γ.

        The default dispatches on the op's declared block-execution
        trait from ``builtin.dgen``. An op declaring none fails
        verification. A block-holding op with bespoke execution
        semantics may override this and compose the context's
        primitives instead. See ``docs/linear_types.md``.
        """
        ctx.from_traits()
