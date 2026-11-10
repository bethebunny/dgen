"""Tests for layout.Record — fixed struct of named fields."""

from dgen.layout import Array, Byte, Float64, Int, Record, Void


def test_record_byte_size():
    r = Record([("x", Int()), ("y", Float64())])
    assert r.byte_size == Int().byte_size + Float64().byte_size


def test_record_round_trip_dict():
    r = Record([("x", Int()), ("y", Float64())])
    buf = bytearray(r.byte_size)
    origins: list[bytearray] = []
    r.from_json(buf, 0, {"x": 42, "y": 3.14}, origins)
    result = r.to_json(buf, 0)
    assert result == {"x": 42, "y": 3.14}


def test_record_single_field():
    r = Record([("val", Int())])
    buf = bytearray(r.byte_size)
    origins: list[bytearray] = []
    r.from_json(buf, 0, {"val": 99}, origins)
    assert r.to_json(buf, 0) == {"val": 99}


def test_record_with_void():
    r = Record([("x", Int()), ("tag", Void())])
    buf = bytearray(r.byte_size)
    origins: list[bytearray] = []
    r.from_json(buf, 0, {"x": 7, "tag": None}, origins)
    result = r.to_json(buf, 0)
    assert result == {"x": 7, "tag": None}


def test_record_nested():
    inner = Record([("a", Int()), ("b", Int())])
    outer = Record([("first", inner), ("second", Float64())])
    buf = bytearray(outer.byte_size)
    origins: list[bytearray] = []
    outer.from_json(buf, 0, {"first": {"a": 1, "b": 2}, "second": 9.0}, origins)
    result = outer.to_json(buf, 0)
    assert result == {"first": {"a": 1, "b": 2}, "second": 9.0}


def test_record_aligns_fields():
    r = Record([("flag", Byte()), ("x", Int())])
    assert r.offsets == [0, 8]
    assert r.byte_size == 16


def test_record_pads_to_alignment():
    r = Record([("x", Int()), ("flag", Byte())])
    assert r.byte_size == 16
    assert r.alignment == 8


def test_record_of_bytes_is_packed():
    r = Record([("a", Byte()), ("b", Byte())])
    assert r.byte_size == 2
    assert r.alignment == 1


def test_padded_record_roundtrip():
    r = Record([("flag", Byte()), ("x", Int()), ("y", Byte())])
    buf = bytearray(r.byte_size)
    r.from_json(buf, 0, {"flag": 1, "x": -5, "y": 2}, [])
    assert r.to_json(buf, 0) == {"flag": 1, "x": -5, "y": 2}


def test_array_of_padded_records_strides_by_size():
    r = Record([("x", Int()), ("flag", Byte())])
    a = Array(r, 2)
    buf = bytearray(a.byte_size)
    value = [{"x": 1, "flag": 1}, {"x": 2, "flag": 0}]
    a.from_json(buf, 0, value, [])
    assert a.byte_size == 32
    assert a.to_json(buf, 0) == value


def test_layouts_are_little_endian():
    buf = bytearray(8)
    Int().from_json(buf, 0, 1, [])
    assert buf == b"\x01" + bytes(7)
