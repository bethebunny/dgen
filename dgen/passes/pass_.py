"""Pass base class for the pass infrastructure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import dgen
from dgen.graph import all_blocks
from dgen.module import Module
from dgen.verify import (
    verify_all_ready,
    verify_closed_blocks,
    verify_constraints,
    verify_dag,
)

if TYPE_CHECKING:
    from dgen.compiler import Compiler

# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

_HandlerFn = Callable[..., dgen.Value | None]


def lowering_for(value_type: type[dgen.Value]) -> Callable[[_HandlerFn], _HandlerFn]:
    """Decorator to register a method as a handler for a value type.

    Handlers return ``Value | None``:
    - ``Value``: the framework calls ``block.replace_uses_of(old, result)``
    - ``None``: no match, try the next handler
    """

    def decorator(fn: _HandlerFn) -> _HandlerFn:
        if not hasattr(fn, "_lowering_for_ops"):
            fn._lowering_for_ops = []  # type: ignore[attr-defined]
        fn._lowering_for_ops.append(value_type)  # type: ignore[attr-defined]
        return fn

    return decorator


class _PassMeta(type):
    """Metaclass that collects @lowering_for handlers into _handlers."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
    ) -> _PassMeta:
        cls = super().__new__(mcs, name, bases, namespace)
        # Collect handlers from all bases + this class
        handlers: dict[type[dgen.Value], list[_HandlerFn]] = defaultdict(list)
        for base in bases:
            if hasattr(base, "_handlers"):
                for value_type, fns in base._handlers.items():
                    handlers[value_type].extend(fns)
        # Add handlers from this class
        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and hasattr(attr_value, "_lowering_for_ops"):
                for value_type in attr_value._lowering_for_ops:
                    handlers[value_type].append(attr_value)
        cls._handlers = dict(handlers)  # type: ignore[attr-defined]
        return cls  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Pass base class
# ---------------------------------------------------------------------------


class Pass(metaclass=_PassMeta):
    """Base class for IR passes.

    Subclasses register ``@lowering_for`` handlers that return
    ``Value | None``. The framework calls ``block.replace_uses_of``
    automatically when a handler returns a value.
    """

    _handlers: dict[type[dgen.Value], list[_HandlerFn]]

    allow_unregistered_ops: bool = True

    def _dispatch_handlers(self, v: dgen.Value) -> dgen.Value | None:
        """Try handlers for v, return replacement or None."""
        for handler in self._handlers.get(type(v), []):
            if (result := handler(self, v)) is not None:
                return result
        if not self.allow_unregistered_ops and isinstance(v, dgen.Op):
            raise TypeError(
                f"No handler for {type(v).__name__} in {type(self).__name__}"
            )
        return None

    def run(self, value: dgen.Value, compiler: Compiler[object]) -> dgen.Value:
        """Run this pass on a value and all its nested blocks."""
        root_block = dgen.Block(result=value)
        for block in [root_block, *all_blocks(value)]:
            for v in list(block.values):
                if (result := self._dispatch_handlers(v)) is not None:
                    block.replace_uses_of(v, result)
        return root_block.result

    def verify_preconditions(self, module: Module) -> None:
        """Check IR invariants that must hold before this pass runs."""
        verify_all_ready(module)
        verify_dag(module)
        verify_closed_blocks(module)
        verify_constraints(module)

    def verify_postconditions(self, module: Module) -> None:
        """Check IR invariants that must hold after this pass runs."""
        verify_dag(module)
        verify_closed_blocks(module)
