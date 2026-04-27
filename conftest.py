"""Root conftest.py — shared pytest fixtures."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion

import dgen
from dgen.asm import asm_with_imports
from dgen.asm.parser import parse
from dgen.passes.compiler import Compiler, IdentityPass, verify_passes
from dgen.passes.pass_ import Pass
from dgen.testing import strip_prefix
from dgen.testing.syrupy import IRSnapshotExtension, LoweringSnapshot

# Make toy and dcc dialects discoverable via dgen.imports.load().
dgen.PATH.append(Path(__file__).parent / "examples" / "toy" / "dialects")
dgen.PATH.append(Path(__file__).parent / "examples" / "dcc" / "dialects")


@pytest.fixture(autouse=True)
def _enable_pass_verification():
    """Enable IR verification for all pass pre/postconditions in every test."""
    token = verify_passes.set(True)
    yield
    verify_passes.reset(token)


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


def _has_pipe_prefix(text: str) -> bool:
    """True if the text uses the ``| ...`` pipe-prefix convention."""
    return any(line.lstrip().startswith("|") for line in text.splitlines())


@pytest.fixture
def lowering_snapshot(
    ir_snapshot: SnapshotAssertion,
) -> Callable[[Sequence[Pass], str], dgen.Value]:
    """Run passes on input IR and snapshot the result with a provenance header.

    Returns a callable ``(passes, ir_text) -> Value`` that:

    1. Parses ``ir_text`` (auto-handles ``strip_prefix``-style ``| `` prefixes).
    2. Captures the input as canonical IR text.
    3. Runs ``Compiler(passes, IdentityPass()).run(...)``.
    4. Asserts the result against ``ir_snapshot``, with a leading ``#`` comment
       block in the snapshot file naming the passes and embedding the input IR.
    5. Returns the lowered ``Value`` so the caller can do additional assertions.

    Example::

        def test_break_lowering(lowering_snapshot):
            result = lowering_snapshot([ControlFlowToGoto()], '''
                | import control_flow
                | %brk : Never = control_flow.break()
            ''')
            for v in all_values(result):
                assert not isinstance(v, control_flow.BreakOp)
    """

    def assert_lowering(passes: Sequence[Pass], ir_text: str) -> dgen.Value:
        passes = list(passes)
        normalized = strip_prefix(ir_text) if _has_pipe_prefix(ir_text) else ir_text
        input_value = parse(normalized)
        input_asm = "\n".join(asm_with_imports(input_value)).rstrip("\n")
        result = Compiler(passes, IdentityPass()).run(input_value)
        assert (
            LoweringSnapshot(
                result=result,
                pass_names=tuple(type(p).__name__ for p in passes),
                input_asm=input_asm,
            )
            == ir_snapshot
        )
        return result

    return assert_lowering
