"""Tests for the wire format: Memory.to_wire() and Memory.from_wire()."""

import subprocess
import sys
from pathlib import Path

import pytest

from dgen import layout
from dgen.asm.parser import parse
from dgen.dialects import builtin, function, ndbuffer, number
from dgen.dialects.builtin import Nil, String, Tuple
from dgen.dialects.index import Index
from dgen.module import pack
from dgen.testing import llvm_compile, strip_prefix
from dgen.type import (
    WIRE_HEADER,
    WIRE_MAGIC,
    Memory,
    Type,
    TypeType,
    format_value,
    wire_json,
)
from toy.dialects import toy

REPO = Path(__file__).parent.parent


def span(pointee: Type) -> builtin.Span:
    return builtin.Span(pointee=pointee)


def array(element_type: Type, n: int) -> builtin.Array:
    return builtin.Array(element_type=element_type, n=Index().constant(n))


def shape(rank: int) -> ndbuffer.Shape:
    return ndbuffer.Shape(rank=Index().constant(rank))


VALUES: list[tuple[Type, object]] = [
    (Nil(), None),
    (Index(), -42),
    (number.Float64(), 2.5),
    (number.Boolean(), 1),
    (builtin.Byte(), 255),
    (number.SignedInteger(bits=Index().constant(32)), 7),
    (String(), "héllo"),
    (String(), ""),
    (builtin.Pointer(pointee=Index()), 9),
    (builtin.Pointer(pointee=String()), "pointee"),
    (array(Index(), 3), [1, 2, 3]),
    (array(span(Index()), 3), [[1], [2, 3], []]),
    (array(array(number.Float64(), 2), 2), [[1.0, 2.0], [3.0, 4.0]]),
    (span(Index()), [10, 20, 30]),
    (span(Index()), []),
    (span(span(Index())), [[1, 2], [], [3]]),
    (span(span(String())), [["a", "bc"], [""]]),
    (span(TypeType()), [{"tag": "index.Index"}, {"tag": "builtin.String"}]),
    (Tuple(types=pack([Index(), String()])), {"0": 42, "1": "hello"}),
    (Tuple(types=pack([number.Boolean(), Index()])), {"0": 1, "1": -1}),
    (Tuple(types=pack([])), {}),
    (Tuple(types=pack([span(String()), array(Index(), 2)])), {"0": ["x"], "1": [4, 5]}),
    (shape(2), [2, 3]),
    (TypeType(), {"tag": "index.Index"}),
    (TypeType(), {"tag": "builtin.Span", "pointee": {"tag": "index.Index"}}),
    (TypeType(), {"tag": "ndbuffer.Shape", "rank": 2}),
    (
        TypeType(),
        {
            "tag": "builtin.Tuple",
            "types": [{"tag": "index.Index"}, {"tag": "builtin.String"}],
        },
    ),
    (toy.Tensor(shape=shape(2).constant([2, 3]), dtype=number.Float64()), [1.0] * 6),
]


@pytest.mark.parametrize(
    ("type", "value"), VALUES, ids=[format_value(t) for t, _ in VALUES]
)
def test_wire_roundtrip(type: Type, value: object):
    memory = Memory.from_json(type, value)
    loaded = Memory.from_wire(memory.to_wire())
    assert loaded.to_json() == memory.to_json()
    assert format_value(loaded.type) == format_value(type)
    assert loaded.to_wire() == memory.to_wire()


@pytest.mark.parametrize(
    ("type", "value"), VALUES, ids=[format_value(t) for t, _ in VALUES]
)
def test_wire_is_deterministic(type: Type, value: object):
    assert (
        Memory.from_json(type, value).to_wire()
        == Memory.from_json(type, value).to_wire()
    )


def test_wire_header():
    data = Memory.from_json(Index(), 5).to_wire()
    magic, version, type_at, value, relocations, count = WIRE_HEADER.unpack_from(data)
    assert (magic, version) == (WIRE_MAGIC, 1)
    assert data[value : value + 8] == (5).to_bytes(8, "little")
    assert relocations % 8 == 0
    assert len(data) == relocations + 8 * count
    assert type_at > value


def test_wire_has_no_host_addresses():
    """Every relocated slot holds an offset inside the image."""
    data = Memory.from_json(span(span(String())), [["a"], ["b", "c"]]).to_wire()
    *_, relocations, count = WIRE_HEADER.unpack_from(data)
    slots = [
        int.from_bytes(data[at : at + 8], "little")
        for at in range(relocations, relocations + 8 * count, 8)
    ]
    targets = [int.from_bytes(data[slot : slot + 8], "little") for slot in slots]
    assert all(WIRE_HEADER.size <= target < relocations for target in targets)


def test_wire_loads_into_one_buffer():
    memory = Memory.from_wire(
        Memory.from_json(span(span(Index())), [[1], [2, 3]]).to_wire()
    )
    (image,) = memory.origins
    start = layout._bytearray_address(image)
    address, _ = memory.unpack()
    assert start <= address < start + len(image)


def test_wire_zeroes_record_padding():
    tuple_type = Tuple(types=pack([number.Boolean(), Index()]))
    memory = Memory.from_json(tuple_type, {"0": 1, "1": 2})
    memory.buffer[1:8] = b"\xff" * 7
    assert memory.to_wire() == Memory.from_json(tuple_type, {"0": 1, "1": 2}).to_wire()


def test_wire_rejects_host_addresses():
    func = function.Function(arguments=pack([]), result_type=Nil())
    memory = Memory(func)
    memory.buffer[:] = (1234).to_bytes(8, "little")
    with pytest.raises(ValueError, match="host address"):
        memory.to_wire()


def test_wire_rejects_other_bytes():
    with pytest.raises(ValueError, match="not a dgen wire image"):
        Memory.from_wire(b"\0" * WIRE_HEADER.size)


def test_memory_equality_is_bitwise():
    assert Memory.from_json(number.Float64(), 0.0) != Memory.from_json(
        number.Float64(), -0.0
    )
    nan = Memory.from_json(number.Float64(), float("nan"))
    assert nan == Memory.from_json(number.Float64(), float("nan"))


def test_memory_equality_includes_type():
    assert Memory.from_json(Index(), 1) != Memory.from_json(
        number.SignedInteger(bits=Index().constant(64)), 1
    )


def test_layout_descriptor_parses_to_layout():
    for type, _ in VALUES:
        assert layout.parse(str(type.__layout__)) == type.__layout__


def test_type_record_names_layout():
    record = Memory.from_wire(span(String()).__constant__.to_wire())
    assert record.to_json() == {
        "tag": "builtin.Span",
        "pointee": {"tag": "builtin.String"},
    }


def test_loaded_type_value_passes_through_jit():
    ir = strip_prefix("""
        | import function
        |
        | %main : function.Function<[Type], Type> = function.function<Type>() body(%t: Type):
        |     %_ : Type = chain(%t, ())
    """)
    exe = llvm_compile(parse(ir))
    loaded = Memory.from_wire(span(Index()).__constant__.to_wire())
    assert exe.run(loaded).to_json() == {
        "tag": "builtin.Span",
        "pointee": {"tag": "index.Index"},
    }


WRITER = """
import sys
from dgen.dialects.builtin import Index, Span, String
from dgen.type import Memory
memory = Memory.from_json(Span(pointee=Span(pointee=String())), [["one", "two"], [], ["three"]])
sys.stdout.write(memory.to_wire().hex())
"""

READER = """
import sys
from toy.dialects import toy
from dgen.type import Memory
print(Memory.from_wire(bytes.fromhex(sys.stdin.read())).to_json())
"""

# Reads without importing toy, so the dialect is not registered.
GENERIC_READER = """
import sys
from dgen.type import wire_json
print(wire_json(bytes.fromhex(sys.stdin.read())))
"""


def run_python(code: str, stdin: str = "") -> str:
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        input=stdin,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_wire_crosses_processes():
    data = bytes.fromhex(run_python(WRITER))
    loaded = Memory.from_wire(data)
    assert loaded.to_json() == [["one", "two"], [], ["three"]]
    assert format_value(loaded.type) == "Span<Span<String>>"


def test_wire_crosses_processes_back():
    tensor = toy.Tensor(shape=shape(2).constant([2, 2]), dtype=number.Float64())
    data = Memory.from_json(tensor, [1.0, 2.0, 3.0, 4.0]).to_wire()
    assert run_python(READER, data.hex()) == "[1.0, 2.0, 3.0, 4.0]\n"


def test_wire_reads_without_the_dialect():
    tensor = toy.Tensor(shape=shape(2).constant([1, 2]), dtype=number.Float64())
    data = Memory.from_json(tensor, [1.0, 2.0]).to_wire()
    type_json = {
        "tag": "toy.Tensor",
        "shape": [1, 2],
        "dtype": {"tag": "number.Float64"},
    }
    assert run_python(GENERIC_READER, data.hex()) == f"{(type_json, [1.0, 2.0])}\n"


def test_wire_json():
    data = Memory.from_json(span(String()), ["a", "b"]).to_wire()
    assert wire_json(data) == (
        {"tag": "builtin.Span", "pointee": {"tag": "builtin.String"}},
        ["a", "b"],
    )
