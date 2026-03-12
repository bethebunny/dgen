"""Root conftest.py — shared pytest fixtures."""

import pytest
from syrupy.assertion import SnapshotAssertion

from dgen.testing.syrupy import IRSnapshotExtension


@pytest.fixture
def ir_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Snapshot fixture using IR graph-equivalence comparison.

    Snapshots are stored as ``.ir`` text files in a ``__snapshots__``
    subdirectory next to the test file.  Comparison is order- and
    label-insensitive: the test passes if the two modules are graph-equivalent,
    regardless of op ordering or SSA name choice.

    Example::

        def test_my_pass(ir_snapshot):
            result = my_pass(module)
            assert result == ir_snapshot

    Run ``pytest --snapshot-update`` to generate or update snapshots.
    """
    return snapshot.use_extension(IRSnapshotExtension)
