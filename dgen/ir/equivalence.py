"""IR graph equivalence via Merkle fingerprinting.

Two IRs are equivalent if their use-def graphs are structurally isomorphic
— same computation up to op ordering and alpha-renaming.

See docs/ir_testing.md for design rationale.
"""

from __future__ import annotations

import hashlib
import json
import struct

import dgen
from dgen.block import Block, BlockArgument, BlockParameter
from dgen.dialects.builtin import Nil
from dgen.ir.traversal import all_values
from dgen.builtins import PackOp
from dgen.type import Constant, Type, Value


def _hash_parts(*parts: bytes) -> bytes:
    """Hash an ordered sequence of byte strings into a single digest."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(struct.pack(">I", len(part)))
        hasher.update(part)
    return hasher.digest()


class Fingerprinter:
    """Computes content-addressed fingerprints for IR values.

    Fingerprints are keyed on object identity and memoized. A single
    Fingerprinter instance should be used for one coherent IR graph.
    Block arguments must be registered (via register_block) before
    fingerprinting any op that uses them.
    """

    def __init__(self) -> None:
        self._cache: dict[Value, bytes] = {}
        self._arg_positions: dict[BlockArgument | BlockParameter, int] = {}

    def register_block(self, block: Block) -> None:
        """Register block argument/parameter positions and recurse into nested blocks."""
        for i, val in enumerate([*block.parameters, *block.args]):
            self._arg_positions[val] = i
        for v in block.values:
            if isinstance(v, Block):
                self.register_block(v)
            elif isinstance(v, dgen.Op):
                for _, nested in v.blocks:
                    self.register_block(nested)

    def fingerprint(self, value: Value) -> bytes:
        if value in self._cache:
            return self._cache[value]
        result = self._compute(value)
        self._cache[value] = result
        return result

    def _fingerprint_type(self, type_value: Type) -> bytes:
        """Fingerprint a Type by recursing on its structure, bypassing Memory."""
        parts: list[bytes] = [
            b"type",
            type_value.dialect.name.encode(),
            type_value.asm_name.encode(),
        ]
        for _, param in type_value.parameters:
            parts.append(self._fingerprint_type_param(param))
        return _hash_parts(*parts)

    def _fingerprint_type_param(self, value: Value) -> bytes:
        if isinstance(value, Type):
            return self._fingerprint_type(value)
        return self.fingerprint(value)

    def _compute(self, value: Value) -> bytes:
        match value:
            case BlockArgument(type=arg_type) | BlockParameter(type=arg_type):
                pos = self._arg_positions[value]
                return _hash_parts(
                    b"arg", pos.to_bytes(4, "big"), self._fingerprint_type(arg_type)
                )
            case Constant():
                serialized = json.dumps(value.__constant__.to_json(), sort_keys=True)
                return _hash_parts(
                    b"constant", self._fingerprint_type(value.type), serialized.encode()
                )
            case PackOp() as pack:
                element_fingerprints = b"".join(self.fingerprint(v) for v in pack)
                return _hash_parts(b"pack", element_fingerprints)
            case list() as lst:
                element_fingerprints = b"".join(
                    self.fingerprint(v)
                    if isinstance(v, Value)
                    else json.dumps(v).encode()
                    for v in lst
                )
                return _hash_parts(b"list", element_fingerprints)
            case Type() as type_value:
                return self._fingerprint_type(type_value)
            case Block() as block:
                return self._fingerprint_block(block)
            case dgen.Op() as op:
                param_fingerprints = b"".join(
                    self.fingerprint(v) for _, v in op.parameters
                )
                operand_fingerprints = b"".join(
                    self.fingerprint(v) for _, v in op.operands
                )
                return _hash_parts(
                    op.dialect.name.encode(),
                    op.asm_name.encode(),
                    self._fingerprint_type(op.type),
                    param_fingerprints,
                    operand_fingerprints,
                )
            case Value(type=Nil()):
                return _hash_parts(b"nil")
            case _:
                raise TypeError(f"Cannot fingerprint {type(value).__name__}")

    def _fingerprint_block(self, block: Block) -> bytes:
        return _hash_parts(b"block", self.fingerprint(block.result))


def fingerprint(root: Value) -> bytes:
    """Fingerprint a Value's use-def graph.

    Pre-registers every block reachable from root (including captures,
    nested blocks, and cross-value references) so arg/param positions are
    known before any recursive fingerprinting begins.
    """
    fp = Fingerprinter()
    for v in all_values(root):
        if isinstance(v, Block):
            fp.register_block(v)
        elif isinstance(v, dgen.Op):
            for _, block in v.blocks:
                fp.register_block(block)
    return fp.fingerprint(root)


def graph_equivalent(actual: Value, expected: Value) -> bool:
    """Return True if actual and expected have structurally equivalent
    use-def graphs — same ops, same operand structure, up to op ordering
    and SSA name assignment.
    """
    return fingerprint(actual) == fingerprint(expected)
