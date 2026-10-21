#!/usr/bin/env bash
# Runs the demo, the runtime tests (at -O0 and default opt), and checks that
# each compile_fail/*.mojo fails with its EXPECT-ERROR text.
set -u
cd "$(dirname "$0")"
status=0
run() { mojo run -I . "$@" 2>&1 | grep -v "warning:\|^Imported\|^ \|ERROR file_io"; }
echo "== demo";              run demo.mojo || status=1
for opt in -O0 ""; do
    echo "== tests ${opt:-(default opt)}"
    out=$(run $opt test_existential.mojo)
    echo "$out" | tail -1
    echo "$out" | grep -q "all tests passed; live instances: 0" || status=1
done
echo "== generated adapters"
if python3 gen/gen_adapters.py gen/shapes.mojo 2>/dev/null | diff -q - gen/dyn_shapes.mojo >/dev/null; then
    echo "gen/dyn_shapes.mojo is up to date"
else
    echo "gen/dyn_shapes.mojo is STALE: regenerate with gen/gen_adapters.py"; status=1
fi
if python3 gen/gen_adapters.py gen/edge_cases.mojo 2>&1 >/dev/null | diff -q - gen/edge_cases.expected_skips >/dev/null; then
    echo "generator edge-case skips as expected"
else
    echo "generator edge-case skips CHANGED"; status=1
fi
out=$(mojo run -I . -I gen gen/test_gen.mojo 2>&1 | grep -v "warning:\|^Imported\|^ \|ERROR file_io")
echo "$out" | tail -1
echo "$out" | grep -q "all checks passed" || status=1
for f in compile_fail/*.mojo; do
    want=$(sed -n 's/^# EXPECT-ERROR: //p' "$f")
    if mojo run -I . "$f" 2>&1 | grep -q "$want"; then
        echo "== $f: rejected as expected ($want)"
    else
        echo "== $f: UNEXPECTEDLY ACCEPTED or wrong error"; status=1
    fi
done
exit $status
