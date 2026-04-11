"""Dialect package — empty marker plus a few hand-written extensions
that augment generated dialect classes with behaviour the .dgen syntax
can't yet express.
"""

from __future__ import annotations

# ``has trait bound`` for the ``Some`` existential type. The dgen spec
# syntax supports static ``has trait <name>`` declarations on a type, but
# the trait name has to resolve to a class — there is no syntax for "the
# trait is whichever class my ``bound`` parameter holds at instantiation
# time". Until that lands (TODO in TODO.md), monkey-patch ``has_trait``
# on the generated ``Some`` class so a trait-bounded existential answers
# the trait check correctly: ``Some<MyTrait>().has_trait(MyTrait)`` is
# ``True``.
from dgen.dialects.existential import Some as _Some
from dgen.type import Value as _Value


def _some_has_trait(self: _Some, trait: type) -> bool:
    return isinstance(self.bound, trait) or _Value.has_trait(self, trait)


_Some.has_trait = _some_has_trait  # type: ignore[method-assign]
