"""Shared test utilities."""

import tempfile
from collections.abc import Sequence

from click.testing import CliRunner

from dgen.testing import strip_prefix as strip_prefix
from toy.cli import main


def run_toy(source: str, *, args: Sequence[object] | None = None) -> str:
    """Write .toy source to a temp file and run via CliRunner."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toy") as f:
        f.write(source)
        f.flush()
        cli_args = [f.name]
        if args:
            cli_args.extend(str(a) if not isinstance(a, str) else a for a in args)
        r = CliRunner().invoke(main, cli_args)
        assert r.exit_code == 0, r.output
        return r.output.strip()
