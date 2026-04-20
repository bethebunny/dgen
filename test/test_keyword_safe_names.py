"""Tests for Python-keyword-safe block/attribute name mapping.

A ``.dgen`` file may name a block ``except`` (to mirror Python/Mojo's
``try``/``except``) or any other keyword-conflicting name. ``except`` is a
Python keyword — it can't appear as a dataclass field or dot-access
attribute. The framework maps the ASM-level name to a Python-safe
attribute by appending an underscore.

Round-trip:

- ``__blocks__`` keeps the original ASM name (``"except"``).
- The dataclass field is ``except_`` (dot-access works).
- ``op.blocks`` yields the original ASM name, looking up through the
  mapping — so asm.format/parse stay unchanged.
- The ASM parser assigns the block via the Python-safe name.
"""

from __future__ import annotations

import dgen.imports
from dgen.block import Block
from dgen.dialects.builtin import Nil
from dgen.testing import strip_prefix
from dgen.type import py_attr_name


def test_py_attr_name_maps_keywords():
    """Python keywords get an underscore suffix; regular names pass through."""
    assert py_attr_name("except") == "except_"
    assert py_attr_name("try") == "try_"
    assert py_attr_name("body") == "body"
    assert py_attr_name("on_raise") == "on_raise"


def test_dgen_block_named_except_compiles():
    """A built op with ``block except`` has ``except_`` as a dataclass field,
    ``"except"`` in ``__blocks__``, and round-trips through ``op.blocks``."""
    dialect = dgen.imports.load(
        "_kwtest",
        source=strip_prefix("""
            | op my_try():
            |     block body
            |     block except
        """),
    )
    my_try = dialect.ops["my_try"]

    # __blocks__ preserves the ASM-level name.
    assert my_try.__blocks__ == ("body", "except")
    # The Python field name is keyword-safe.
    field_names = {f.name for f in my_try.__dataclass_fields__.values()}
    assert "except_" in field_names
    assert "except" not in field_names

    # Construction via the Python-safe keyword argument works.
    body = Block(result=Nil().constant(None))
    except_block = Block(result=Nil().constant(None))
    op = my_try(body=body, except_=except_block, type=Nil())

    # op.except_ dot-access works; op.blocks yields the ASM name.
    assert op.except_ is except_block
    assert list(op.blocks) == [("body", body), ("except", except_block)]
