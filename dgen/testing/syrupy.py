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

For lowering tests, ``LoweringSnapshot`` augments a ``Value`` with provenance
metadata (which passes were run, what the input was) which is emitted as a
leading ``#`` comment block in the snapshot file.  The comment is ignored by
the parser, so equivalence comparison is unaffected; the header is purely for
human readability of the snapshot file.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from syrupy.types import (
    PropertyFilter,
    PropertyMatcher,
    SerializableData,
    SerializedData,
)

from dgen.asm.parser import parse
from dgen.ir.diff import diff_values
from dgen.ir.equivalence import graph_equivalent
from dgen.asm import asm_with_imports
from dgen.type import Value


@dataclass(frozen=True)
class LoweringSnapshot:
    """A lowered IR value plus provenance for the snapshot comment header.

    Tests should not construct this directly — use the ``lowering_snapshot``
    fixture in ``conftest.py``, which builds a ``LoweringSnapshot`` and asserts
    it against ``ir_snapshot``.

    Fields:
      - ``result``: the lowered ``Value`` (this is what equivalence checks see)
      - ``pass_names``: human-readable labels for the passes that produced it
      - ``input_asm``: the formatted input IR, embedded as a comment header
    """

    result: Value
    pass_names: tuple[str, ...]
    input_asm: str


def _format_lowering_header(pass_names: tuple[str, ...], input_asm: str) -> str:
    """Build the leading ``#``-comment block for a ``LoweringSnapshot``."""
    lines: list[str] = []
    if len(pass_names) == 1:
        lines.append(f"# Lowered through: {pass_names[0]}")
    else:
        lines.append(f"# Lowered through {len(pass_names)} passes:")
        for name in pass_names:
            lines.append(f"#   - {name}")
    lines.append("#")
    lines.append("# Input IR:")
    lines.append("#")
    for line in input_asm.splitlines():
        lines.append(f"#   {line}" if line else "#")
    return "\n".join(lines)


class IRSnapshotExtension(SingleFileSnapshotExtension):
    """Syrupy extension for IR ``Value`` snapshot testing.

    Snapshots are stored as plain ``.ir`` text files (one file per test).
    Comparison uses ``graph_equivalent``, so tests remain green across
    semantically-irrelevant reformattings (op reordering, SSA renaming).
    Failure output uses ``diff_values`` for a semantic, noise-free diff.

    Accepts either a ``Value`` (plain snapshot) or a ``LoweringSnapshot``
    (which adds a leading comment block describing the passes and input IR).
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
        if isinstance(data, LoweringSnapshot):
            header = _format_lowering_header(data.pass_names, data.input_asm)
            body = "\n".join(asm_with_imports(data.result))
            return f"{header}\n\n{body}"
        assert isinstance(data, Value), (
            f"IRSnapshotExtension expects a Value or LoweringSnapshot, "
            f"got {type(data).__name__}"
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
