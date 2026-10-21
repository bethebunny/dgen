#!/usr/bin/env bash
# Runs the demo, the runtime tests (at -O0 and default opt), and checks that
# each compile_fail/*.mojo fails with its EXPECT-ERROR text.
set -u
cd "$(dirname "$0")"
status=0
run() { mojo run -I . "$@" 2>&1 | grep -v "warning:\|^Imported\|^ \|ERROR file_io"; }
echo "== demo";              run demo.mojo || status=1
echo "== tests (-O0)";       run -O0 test_existential.mojo | tail -1
echo "== tests (default)";   run test_existential.mojo | tail -1
for f in compile_fail/*.mojo; do
    want=$(sed -n 's/^# EXPECT-ERROR: //p' "$f")
    if mojo run -I . "$f" 2>&1 | grep -q "$want"; then
        echo "== $f: rejected as expected ($want)"
    else
        echo "== $f: UNEXPECTEDLY ACCEPTED or wrong error"; status=1
    fi
done
exit $status
