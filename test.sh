set -ex

ROOT="$(pwd)"
BUILD="$ROOT/build"
SRC="$ROOT/toy"

mkdir -p "$BUILD"
cd "$BUILD"
mojo package "$SRC"
for test in "$SRC/tests/"*; do
    mojo build -I "$BUILD" "$test"
    TEST="$(basename "$test")"
    echo "Testing $TEST"
    ./"${TEST%.mojo}"
done