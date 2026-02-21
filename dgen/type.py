from typing import Protocol


class Type(Protocol):
    """Any dialect type."""

    @property
    def asm(self) -> str: ...
