# App B defines a new trait after the library was written.
from existential import *

trait Describe:
    def describe(self) -> String: ...

@fieldwise_init
struct DescribeVT(ImplicitlyCopyable):
    var describe: def(Opaque) thin -> String

struct DynDescribe(Describe, TraitAdapter):
    comptime Trait = Describe
    comptime VTable = DescribeVT
    var _h: Header
    @staticmethod
    def vtable_for[T: Describe]() -> DescribeVT:
        return DescribeVT(thunk[T.describe])
    def describe(self) -> String:
        return self._h.vtable[Self]().describe(self._h.data)
