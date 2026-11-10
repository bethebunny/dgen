"""Root conftest.py — shared pytest fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion

from dgen import Dialect
from dgen.compiler import verify_passes
from dgen.graph import RandomOrder, random_order
from dgen.testing.syrupy import IRSnapshotExtension

# Make toy and dcc2 dialects discoverable via Dialect.get().
Dialect.paths.append(Path(__file__).parent / "toy" / "dialects")
Dialect.paths.append(Path(__file__).parent / "dcc2" / "dialects")


@pytest.fixture(autouse=True)
def _enable_pass_verification():
    """Enable IR verification for all pass pre/postconditions in every test."""
    token = verify_passes.set(True)
    yield
    verify_passes.reset(token)


@pytest.fixture(autouse=True)
def _walk_order(request: pytest.FixtureRequest) -> Iterator[None]:
    """Walk in a random topological order under --walk-seeds, else depth-first."""
    seed: int | None = request.param if _random_walks(request.node) else None
    token = random_order.set(None if seed is None else RandomOrder(seed))
    yield
    random_order.reset(token)


def _random_walks(item: pytest.Item) -> bool:
    return (
        item.config.getoption("walk_seeds") > 0
        and item.get_closest_marker("canonical_order") is None
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "_walk_order" in metafunc.fixturenames and _random_walks(metafunc.definition):
        seeds = range(metafunc.config.getoption("walk_seeds"))
        metafunc.parametrize(
            "_walk_order", seeds, indirect=True, ids=lambda s: f"walk-seed-{s}"
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--walk-seeds",
        type=int,
        default=0,
        help="Run each test under this many random topological walk orders.",
    )
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
