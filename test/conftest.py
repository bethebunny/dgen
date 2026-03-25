# Ensure dialects are registered before tests parse IR text with `import llvm` etc.
from dgen.dialects import function as _function  # noqa: F401
from dgen.dialects import llvm as _llvm  # noqa: F401
