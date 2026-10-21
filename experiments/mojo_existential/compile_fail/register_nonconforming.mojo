# EXPECT-ERROR: violated constraint
from existential import *

trait Shape:
    def area(self) -> Int: ...

@fieldwise_init
struct ShapeVT(ImplicitlyCopyable):
    var area: def(Opaque) thin -> Int

struct DynShape(Shape, TraitAdapter):
    comptime Trait = Shape
    comptime VTable = ShapeVT
    var _h: Header
    @staticmethod
    def vtable_for[T: Shape]() -> ShapeVT:
        return ShapeVT(thunk[T.area])
    def area(self) -> Int:
        return self._h.vtable[Self]().area(self._h.data)

def main():
    register[Int, DynShape]()  # Int is not a Shape
