"""Benchmark tests for the C frontend."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from dgen_c.parser.c_parser import parse_c_string
from dgen_c.parser.lowering import lower

TESTDATA = Path(__file__).parent / "testdata"


@pytest.mark.skipif(
    not (TESTDATA / "large_sqlite_like.c").exists(),
    reason="large_sqlite_like.c not generated",
)
class TestBenchmark:
    def test_parse_200k_lines(self) -> None:
        """Parse a 200K-line C file in a reasonable time."""
        source = (TESTDATA / "large_sqlite_like.c").read_text()
        t0 = time.perf_counter()
        ast = parse_c_string(source)
        elapsed = time.perf_counter() - t0
        assert elapsed < 60, f"Parsing took {elapsed:.1f}s (limit: 60s)"
        assert len(ast.ext) > 5000

    def test_lower_200k_lines(self) -> None:
        """Lower a 200K-line C file in a reasonable time."""
        source = (TESTDATA / "large_sqlite_like.c").read_text()
        ast = parse_c_string(source)
        t0 = time.perf_counter()
        module, stats = lower(ast)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30, f"Lowering took {elapsed:.1f}s (limit: 30s)"
        assert stats.functions >= 5000
        assert stats.expressions > 300000
        assert stats.skipped_stmts == 0

    def test_full_pipeline_200k_lines(self) -> None:
        """Full parse+lower pipeline on 200K-line C file."""
        source = (TESTDATA / "large_sqlite_like.c").read_text()
        t0 = time.perf_counter()
        ast = parse_c_string(source)
        module, stats = lower(ast)
        elapsed = time.perf_counter() - t0
        assert elapsed < 60, f"Pipeline took {elapsed:.1f}s (limit: 60s)"
        assert len(module.functions) >= 5000
