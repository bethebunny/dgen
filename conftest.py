"""Root conftest.py — shared pytest fixtures."""

from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion

from dgen import Dialect
from dgen.asm.parser import verify_parsed
from dgen.compiler import verify_passes
from dgen.testing.syrupy import IRSnapshotExtension

# Make toy and dcc2 dialects discoverable via Dialect.get().
Dialect.paths.append(Path(__file__).parent / "toy" / "dialects")
Dialect.paths.append(Path(__file__).parent / "dcc2" / "dialects")


@pytest.fixture(autouse=True)
def _enable_verification():
    """Verify every parsed value, and every pass's pre and postconditions, in every test."""
    passes = verify_passes.set(True)
    parsed = verify_parsed.set(True)
    yield
    verify_parsed.reset(parsed)
    verify_passes.reset(passes)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--side-by-side",
        action="store_true",
        default=False,
        help="Show IR snapshot diffs side-by-side via delta (requires git-delta).",
    )


@pytest.fixture
def ir_snapshot(
    snapshot: SnapshotAssertion, request: pytest.FixtureRequest
) -> SnapshotAssertion:
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
    IRSnapshotExtension.side_by_side = request.config.getoption("--side-by-side")
    return snapshot.use_extension(IRSnapshotExtension)
