"""Core dgen lowering passes."""

from __future__ import annotations

import dgen
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.passes.existential_to_record import ExistentialToRecord
from dgen.passes.ndbuffer_to_memory import NDBufferToMemory
from dgen.passes.normalize_region_terminators import NormalizeRegionTerminators
from dgen.passes.raise_catch_to_goto import RaiseCatchToGoto
from dgen.passes.record_to_memory import RecordToMemory
from dgen.passes.unpack_to_goto import UnpackToGoto


def lower_builtin_dialects() -> Compiler[dgen.Value]:
    """Core dgen dialect lowerings: control flow, raise/catch, ndbuffer, existential, record."""
    return Compiler(
        passes=[
            ControlFlowToGoto(),
            RaiseCatchToGoto(),
            UnpackToGoto(),
            NormalizeRegionTerminators(),
            NDBufferToMemory(),
            ExistentialToRecord(),
            RecordToMemory(),
        ],
        exit=IdentityPass(),
    )
