"""Syrupy snapshot extension for IR graph equivalence testing.

Provides ``IRSnapshotExtension``, a syrupy extension that stores IR value
snapshots as ``.ir`` text files and compares them using graph equivalence
(order- and label-insensitive Merkle fingerprinting) rather than string equality.

Usage
-----
In ``conftest.py``::

    import pytest
    from dgen.testing.syrupy import IRSnapshotExtension

    @pytest.fixture
    def ir_snapshot(snapshot):
        return snapshot.use_extension(IRSnapshotExtension)

In tests::

    def test_my_pass(ir_snapshot):
        result = my_pass(value)
        assert result == ir_snapshot

On the first run (no snapshot file yet), pass ``--snapshot-update`` to generate
the initial snapshot.  Subsequent runs compare using ``graph_equivalent``; the
test passes as long as the computation is semantically identical, regardless of
op ordering or SSA name choice.

When the output changes semantically, run ``--snapshot-update`` again to accept
the new snapshot.  The failure output shows a semantic diff via ``diff_values``
rather than a raw text diff, so only real changes are reported.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from typing import Optional

from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from syrupy.types import (
    PropertyFilter,
    PropertyMatcher,
    SerializableData,
    SerializedData,
)

from dgen.asm.parser import parse
from dgen.ir_diff import diff_values
from dgen.ir_equiv import graph_equivalent
from dgen.module import asm_with_imports
from dgen.type import Value


class IRSnapshotExtension(SingleFileSnapshotExtension):
    """Syrupy extension for IR ``Value`` snapshot testing.

    Snapshots are stored as plain ``.ir`` text files (one file per test).
    Comparison uses ``graph_equivalent``, so tests remain green across
    semantically-irrelevant reformattings (op reordering, SSA renaming).
    Failure output uses ``diff_values`` for a semantic, noise-free diff.
    """

    file_extension = "ir"
    _write_mode = WriteMode.TEXT
    side_by_side: bool = False

    def serialize(
        self,
        data: SerializableData,
        *,
        exclude: Optional[PropertyFilter] = None,
        include: Optional[PropertyFilter] = None,
        matcher: Optional[PropertyMatcher] = None,
    ) -> SerializedData:
        assert isinstance(data, Value), (
            f"IRSnapshotExtension expects a Value, got {type(data).__name__}"
        )
        return "\n".join(asm_with_imports(data))

    def matches(
        self,
        *,
        serialized_data: SerializedData,
        snapshot_data: SerializedData,
    ) -> bool:
        assert isinstance(serialized_data, str)
        assert isinstance(snapshot_data, str)
        actual = parse(serialized_data)
        expected = parse(snapshot_data)
        return graph_equivalent(actual, expected)

    def diff_lines(
        self, serialized_data: SerializedData, snapshot_data: SerializedData
    ) -> Iterator[str]:
        assert isinstance(serialized_data, str)
        assert isinstance(snapshot_data, str)
        actual = parse(serialized_data)
        expected = parse(snapshot_data)
        diff: str = diff_values(actual, expected)

        if self.side_by_side:
            result = subprocess.run(
                ["delta", "--side-by-side"], input=diff, capture_output=True, text=True
            )
            diff = result.stdout

        yield from diff.splitlines()
