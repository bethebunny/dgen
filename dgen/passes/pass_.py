"""Pass base class and Rewriter for the pass infrastructure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import dgen
from dgen.module import Module, _walk_all_ops


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

_HandlerFn = Callable[..., bool]


def lowering_for(op_type: type[dgen.Op]) -> Callable[[_HandlerFn], _HandlerFn]:
    """Decorator to register a method as a handler for an op type.

    Multiple handlers per op type are allowed; they are tried in
    registration order until one returns True.
    """

    def decorator(fn: _HandlerFn) -> _HandlerFn:
        if not hasattr(fn, "_lowering_for_ops"):
            fn._lowering_for_ops = []  # type: ignore[attr-defined]
        fn._lowering_for_ops.append(op_type)  # type: ignore[attr-defined]
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
        handlers: dict[type[dgen.Op], list[_HandlerFn]] = defaultdict(list)
        for base in bases:
            if hasattr(base, "_handlers"):
                for op_type, fns in base._handlers.items():
                    handlers[op_type].extend(fns)
        # Add handlers from this class
        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and hasattr(attr_value, "_lowering_for_ops"):
                for op_type in attr_value._lowering_for_ops:
                    handlers[op_type].append(attr_value)
        cls._handlers = dict(handlers)  # type: ignore[attr-defined]
        return cls  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------


class Rewriter:
    """Manages in-place IR mutations during a pass."""

    def __init__(self, block: dgen.Block) -> None:
        self._block = block
        self._op_replacements: dict[int, dgen.Op] = {}

    def replace_uses(self, old: dgen.Value, new: dgen.Value) -> None:
        """Eagerly replace all references to old with new in the block."""
        for op in self._block.ops:
            for name, operand in op.operands:
                if operand is old:
                    setattr(op, name, new)
            for name, param in op.parameters:
                if param is old:
                    setattr(op, name, new)

    def replace_op(self, old: dgen.Op, new: dgen.Op) -> None:
        """Replace old op with new op: redirect all uses and swap in the block."""
        self.replace_uses(old, new)
        self._op_replacements[id(old)] = new


# ---------------------------------------------------------------------------
# Pass base class
# ---------------------------------------------------------------------------


class Pass(metaclass=_PassMeta):
    """Base class for IR passes.

    Subclasses declare:
      - op_domain / op_range: sets of op types
      - type_domain / type_range: sets of type types
      - allow_unregistered_ops: bool
      - @lowering_for handlers
    """

    _handlers: dict[type[dgen.Op], list[_HandlerFn]]

    op_domain: set[type]
    op_range: set[type]
    type_domain: set[type]
    type_range: set[type]
    allow_unregistered_ops: bool = True

    def run(self, module: Module) -> Module:
        """Run this pass on all functions in the module."""
        for func in module.functions:
            self._run_block(func.body)
        return module

    def _run_block(self, block: dgen.Block) -> None:
        rewriter = Rewriter(block)
        replaced: set[int] = set()
        for op in list(block.ops):  # snapshot — graph may change
            handlers = self._handlers.get(type(op), [])
            handled = False
            for handler in handlers:
                if handler(self, op, rewriter):
                    handled = True
                    replaced.add(id(op))
                    break
            if not handled and not self.allow_unregistered_ops:
                raise TypeError(
                    f"No handler for {type(op).__name__} in {type(self).__name__}"
                )
            if not handled:
                # Recurse into nested blocks for unhandled ops
                for _, child_block in op.blocks:
                    self._run_block(child_block)
        # Apply replacements: swap or remove handled ops
        if replaced:
            new_ops: list[dgen.Op] = []
            for op in block.ops:
                oid = id(op)
                if oid not in replaced:
                    new_ops.append(op)
                elif oid in rewriter._op_replacements:
                    new_ops.append(rewriter._op_replacements[oid])
                # else: op was replaced via replace_uses only — remove it
            block.ops = new_ops

    def verify_preconditions(self, module: Module) -> None:
        """Check that all ops/types are in the declared domain."""
        for func in module.functions:
            for op in _walk_all_ops(func):
                assert type(op) in self.op_domain, (
                    f"Op {type(op).__name__} not in domain of {type(self).__name__}"
                )

    def verify_postconditions(self, module: Module) -> None:
        """Check that all ops/types are in the declared range."""
        for func in module.functions:
            for op in _walk_all_ops(func):
                assert type(op) in self.op_range, (
                    f"Op {type(op).__name__} not in range of {type(self).__name__}"
                )
