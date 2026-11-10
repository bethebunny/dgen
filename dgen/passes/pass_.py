"""Pass base class for the pass infrastructure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Set as AbstractSet
from typing import TYPE_CHECKING, ClassVar

import dgen
from dgen.dialect import Dialect
from dgen.graph import all_values
from dgen.verify import (
    VerificationError,
    verify_all_ready,
    verify_closed_blocks,
    verify_constraints,
    verify_dag,
)

if TYPE_CHECKING:
    from dgen.compiler import Compiler

type Ops = tuple[Dialect | type[dgen.Op], ...]
"""Dialects and op classes, eg. ``(llvm.llvm, ConstantOp)``."""


class LoweringError(VerificationError):
    """A pass left a value it lowers, or built an op it doesn't emit."""


def ops(entries: Ops) -> frozenset[type[dgen.Op]]:
    """The op classes of entries, with each dialect expanded to its ops."""
    return frozenset(
        op
        for entry in entries
        for op in (entry.ops.values() if isinstance(entry, Dialect) else [entry])
    )


def name(cls: type[dgen.Value]) -> str:
    return (
        f"{cls.dialect.name}.{cls.asm_name}"
        if issubclass(cls, dgen.Op)
        else cls.__name__
    )


def value_classes(root: dgen.Value) -> set[type[dgen.Value]]:
    return {type(v) for v in all_values(root)}


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

_HandlerFn = Callable[..., dgen.Value | None]

# Each tagged function's (value class, whether the handler promises to lower it).
_tags: dict[_HandlerFn, list[tuple[type[dgen.Value], bool]]] = defaultdict(list)


def lowering_for(value_type: type[dgen.Value]) -> Callable[[_HandlerFn], _HandlerFn]:
    """Register a handler that lowers every value of value_type it is given.

    Handlers return ``Value | None``:
    - ``Value``: the framework calls ``block.replace_uses_of(old, result)``
    - ``None``: no match, try the next handler

    The pass promises no value of value_type survives it (see ``Pass.eliminated``).
    """

    def decorator(fn: _HandlerFn) -> _HandlerFn:
        _tags[fn].append((value_type, True))
        return fn

    return decorator


def rewrite_for(value_type: type[dgen.Value]) -> Callable[[_HandlerFn], _HandlerFn]:
    """Register a handler that may decline or return its value, eg. a peephole."""

    def decorator(fn: _HandlerFn) -> _HandlerFn:
        _tags[fn].append((value_type, False))
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Pass base class
# ---------------------------------------------------------------------------


class Pass:
    """Base class for IR passes.

    Subclasses register ``@lowering_for`` and ``@rewrite_for`` handlers that
    return ``Value | None``. The framework calls ``block.replace_uses_of``
    automatically when a handler returns a value.

    Invariants, checked by ``Compiler`` when ``verify_passes`` is set:
      - no value of a class in ``eliminated`` survives the pass
      - every op class the pass adds to the graph is in ``emitted``
    """

    _handlers: ClassVar[dict[type[dgen.Value], list[_HandlerFn]]] = {}

    lowers: ClassVar[Ops] = ()
    """Dialects lowered whole, and op classes removed without a handler of their own."""
    emits: ClassVar[Ops] = ()
    """Dialects and op classes the handlers may build."""
    eliminated: ClassVar[frozenset[type[dgen.Value]]] = frozenset()
    """Derived: lowering_for keys, plus lowers with dialects expanded."""
    emitted: ClassVar[frozenset[type[dgen.Op]]] = frozenset()
    """Derived: emits with dialects expanded."""

    allow_unregistered_ops: bool = True

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        bases = [base for base in cls.__bases__ if issubclass(base, Pass)]
        handlers: dict[type[dgen.Value], list[_HandlerFn]] = defaultdict(list)
        for base in bases:
            for value_type, fns in base._handlers.items():
                handlers[value_type].extend(fns)
        lowered = set().union(*(base.eliminated for base in bases))
        for fn in vars(cls).values():
            for value_type, promise in _tags.get(fn, []) if callable(fn) else []:
                handlers[value_type].append(fn)
                if promise:
                    lowered.add(value_type)
        missing = [
            op
            for dialect in cls.lowers
            if isinstance(dialect, Dialect)
            for op in dialect.ops.values()
            if op not in lowered
        ]
        if missing:
            raise TypeError(
                f"{cls.__name__} lowers {', '.join(sorted({op.dialect.name for op in missing}))} "
                f"but has no lowering_for {', '.join(sorted(map(name, missing)))}"
            )
        cls._handlers = dict(handlers)
        cls.eliminated = frozenset(lowered | ops(cls.lowers))
        cls.emitted = ops(cls.emits)

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
        self._lower_block(root_block)
        return root_block.result

    def _lower_block(self, block: dgen.Block) -> None:
        """Lower all ops in block (topo order), recursing into nested blocks.

        For each op, dispatch handlers. After lowering (whether the op was
        replaced or not), recurse into the effective op's blocks — before
        processing any downstream uses. This ensures handler-created blocks
        are lowered before they're referenced elsewhere.
        """
        for v in block.values:
            for _, child_block in ((result := self._dispatch_handlers(v)) or v).blocks:
                self._lower_block(child_block)
            if result is not None:
                block.replace_uses_of(v, result)

    def verify_preconditions(self, value: dgen.Value) -> None:
        """Check IR invariants that must hold before this pass runs."""
        verify_all_ready(value)
        verify_dag(value)
        verify_closed_blocks(value)
        verify_constraints(value)

    def verify_declarations(
        self, before: AbstractSet[type[dgen.Value]], value: dgen.Value
    ) -> None:
        """Check the pass kept its declarations, given the input's value classes."""
        after = value_classes(value)
        if left := after & self.eliminated:
            raise LoweringError(
                f"{type(self).__name__} left {', '.join(sorted(map(name, left)))}, "
                "which it lowers"
            )
        built = {
            cls for cls in after - before if issubclass(cls, dgen.Op)
        } - self.emitted
        if built:
            raise LoweringError(
                f"{type(self).__name__} built {', '.join(sorted(map(name, built)))}, "
                "which it does not declare in emits"
            )

    def verify_postconditions(self, value: dgen.Value) -> None:
        """Check IR invariants that must hold after this pass runs."""
        verify_dag(value)
        verify_closed_blocks(value)
