"""Tests for layout.Record — fixed struct of named fields."""

from dgen.layout import Record, Int, Float64, Void


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
