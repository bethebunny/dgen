"""CLI tool: parse and compile C files through the dgen pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import click

from dcc.parser.c_parser import parse_c_file
from dcc.parser.lowering import lower


@click.command()
@click.argument("source_file", type=click.Path(exists=True))
@click.option("--stats", is_flag=True, help="Print lowering statistics")
@click.option("--dump-ir", is_flag=True, help="Dump the dgen IR")
@click.option("--benchmark", is_flag=True, help="Print timing information")
@click.option(
    "--cpp/--no-cpp",
    default=False,
    help="Run the C preprocessor (requires gcc/cpp)",
)
def main(
    source_file: str,
    stats: bool,
    dump_ir: bool,
    benchmark: bool,
    cpp: bool,
) -> None:
    """Parse a C source file and lower to dgen IR."""
    path = Path(source_file)

    # Parse
    t0 = time.perf_counter()
    if cpp:
        ast = parse_c_file(path, cpp_args=["-E", "-D__attribute__(x)="])
    else:
        ast = parse_c_file(path)
    t_parse = time.perf_counter() - t0

    # Lower
    t1 = time.perf_counter()
    module, lowering_stats = lower(ast)
    t_lower = time.perf_counter() - t1

    if benchmark:
        click.echo(f"Parse:   {t_parse:.3f}s")
        click.echo(f"Lower:   {t_lower:.3f}s")
        click.echo(f"Total:   {t_parse + t_lower:.3f}s")

    if stats:
        click.echo(lowering_stats.summary())
        click.echo(f"IR functions: {len(module.functions)}")
        total_ops = sum(len(f.body.ops) for f in module.functions)
        click.echo(f"IR ops (reachable): {total_ops}")

    if dump_ir:
        for line in module.asm:
            click.echo(line)


if __name__ == "__main__":
    main()
