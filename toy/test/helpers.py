"""Shared test utilities."""

import tempfile
from collections.abc import Sequence

from click.testing import CliRunner

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


def strip_prefix(text: str) -> str:
    """Convert a pipe-prefixed multiline string to plain text.

    Each line is stripped of leading whitespace and then:
      - "| content" becomes "content"
      - "|"         becomes ""  (blank line)
      - other       passed through as-is
    A trailing newline is always appended.
    """
    lines = text.strip().splitlines()
    result = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("| "):
            result.append(stripped[2:])
        elif stripped == "|":
            result.append("")
        else:
            result.append(stripped)
    return "\n".join(result) + "\n"
