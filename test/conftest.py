# Ensure dialects are registered before tests parse IR text.
from dgen.dialects import algebra as _algebra  # noqa: F401
from dgen.dialects import control_flow as _control_flow  # noqa: F401
from dgen.dialects import function as _function  # noqa: F401
from dgen.dialects import goto as _goto  # noqa: F401
from dgen.dialects import llvm as _llvm  # noqa: F401
from dgen.dialects import memory as _memory  # noqa: F401
from dgen.dialects import ndbuffer as _ndbuffer  # noqa: F401
from dgen.dialects import number as _number  # noqa: F401
