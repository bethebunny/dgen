"""Dialect package — empty marker plus a few hand-written extensions
that augment generated dialect classes with behaviour the .dgen syntax
can't yet express.
"""

from __future__ import annotations

from struct import Struct

from dgen.dialects.existential import PackOp as _PackOp
from dgen.dialects.existential import Some as _Some
from dgen.layout import TypeValue as _TypeValue
from dgen.memory import Memory as _Memory
from dgen.type import Value as _Value
from dgen.type import constant as _constant

# ---------------------------------------------------------------------------
# ``has trait bound`` projection for ``existential.Some``.
#
# The dgen spec syntax supports static ``has trait <name>`` declarations
# on a type, but the trait name has to resolve to a class — there is no
# syntax for "the trait is whichever class my ``bound`` parameter holds
# at instantiation time". Until that lands (TODO in TODO.md),
# monkey-patch ``has_trait`` on the generated ``Some`` class so a
# trait-bounded existential answers the trait check correctly:
# ``Some<MyTrait>().has_trait(MyTrait)`` is ``True``.
# ---------------------------------------------------------------------------


def _some_has_trait(self: _Some, trait: type) -> bool:
    return isinstance(self.bound, trait) or _Value.has_trait(self, trait)


_Some.has_trait = _some_has_trait  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Compile-time constant folding for ``existential.PackOp``.
#
# When the operand of ``pack<bound>(value)`` has a materialisable
# ``__constant__``, the whole pack folds to a constant ``Some<bound>``.
# We can't express this through the layout system because the value
# slot is a ``memory.Reference<Byte>`` (intentionally opaque) — Record's
# generic ``from_json`` would just store a pointer to a 0-byte buffer.
# Instead, manually allocate a 16-byte ``Some`` buffer and write the
# two pointers ourselves: a ``TypeValue`` descriptor for the witness at
# offset 0, and a raw pointer at offset 8 into the inner constant's
# Memory buffer (kept alive via ``mem.origins``).
#
# Runtime ``pack`` is a separate concern handled by the upcoming
# ``existential_to_llvm`` codegen pass — that path emits an alloca and
# memcpy rather than reusing the inner Memory's address.
# ---------------------------------------------------------------------------


_POINTER = Struct("P")


def _pack_constant(self: _PackOp) -> _Memory:
    bound = _constant(self.bound)
    witness = _constant(self.value.type)
    inner = self.value.__constant__
    result_type = _Some(bound=bound)
    mem: _Memory = _Memory(result_type)
    # existential field: TypeValue descriptor for the witness type, at offset 0.
    _TypeValue().from_json(mem.buffer, 0, witness, mem.origins)
    # value field: raw pointer to the inner Memory's bytes, at offset 8.
    # Keep ``inner`` alive via ``origins`` so the buffer doesn't get freed.
    mem.origins.append(inner)
    _POINTER.pack_into(mem.buffer, 8, inner.address)
    return mem


_PackOp.__constant__ = property(_pack_constant)  # type: ignore[method-assign]
