"""Syrupy snapshot extension for IR graph equivalence testing.

Provides ``IRSnapshotExtension``, a syrupy extension that stores IR module
snapshots as ``.ir`` text files and compares them using graph equivalence
(order- and label-insensitive Merkle fingerprinting) rather than string equality.

Usage
-----
In ``conftest.py``::

    import pytest
    from dgen.syrupy_ext import IRSnapshotExtension

    @pytest.fixture
    def ir_snapshot(snapshot):
        return snapshot.use_extension(IRSnapshotExtension)

In tests::

    def test_my_pass(ir_snapshot):
        result = my_pass(module)
        assert result == ir_snapshot

On the first run (no snapshot file yet), pass ``--snapshot-update`` to generate
the initial snapshot.  Subsequent runs compare using ``graph_equivalent``; the
test passes as long as the computation is semantically identical, regardless of
op ordering or SSA name choice.

When the output changes semantically, run ``--snapshot-update`` again to accept
the new snapshot.  The failure output shows a semantic diff via ``diff_modules``
rather than a raw text diff, so only real changes are reported.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from syrupy.types import (
    PropertyFilter,
    PropertyMatcher,
    SerializableData,
    SerializedData,
)

from dgen.asm.parser import parse_module
from dgen.ir_diff import diff_modules
from dgen.ir_equiv import graph_equivalent
from dgen.module import Module


class IRSnapshotExtension(SingleFileSnapshotExtension):
    """Syrupy extension for IR ``Module`` snapshot testing.

    Snapshots are stored as plain ``.ir`` text files (one file per test).
    Comparison uses ``graph_equivalent``, so tests remain green across
    semantically-irrelevant reformattings (op reordering, SSA renaming).
    Failure output uses ``diff_modules`` for a semantic, noise-free diff.
    """

    file_extension = "ir"
    _write_mode = WriteMode.TEXT

    def __init__(self) -> None:
        super().__init__()
        self._current_module: Module | None = None

    def serialize(
        self,
        data: SerializableData,
        *,
        exclude: Optional[PropertyFilter] = None,
        include: Optional[PropertyFilter] = None,
        matcher: Optional[PropertyMatcher] = None,
    ) -> SerializedData:
        assert isinstance(data, Module), (
            f"IRSnapshotExtension expects a Module, got {type(data).__name__}"
        )
        self._current_module = data
        return "\n".join(data.asm)

    def matches(
        self,
        *,
        serialized_data: SerializedData,
        snapshot_data: SerializedData,
    ) -> bool:
        # _current_module is always set by serialize(), which syrupy calls
        # immediately before matches() within the same assertion.
        assert self._current_module is not None
        assert isinstance(snapshot_data, str)
        expected = parse_module(snapshot_data)
        return graph_equivalent(self._current_module, expected)

    def diff_lines(
        self, serialized_data: SerializedData, snapshot_data: SerializedData
    ) -> Iterator[str]:
        # diff_lines is called during failure reporting, after the assertion
        # has completed, so _current_module may be stale. Re-parse from the
        # serialized string, which is stored by syrupy in the assertion result.
        assert isinstance(serialized_data, str)
        assert isinstance(snapshot_data, str)
        actual = parse_module(serialized_data)
        expected = parse_module(snapshot_data)
        yield from diff_modules(actual, expected).splitlines()
