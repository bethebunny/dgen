# EXPECT-ERROR: cannot implicitly convert
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
        def area(p: Opaque) -> Int:
            return p.unsafe_bitcast[T]()[].area()
        return ShapeVT(area)
    def area(self) -> Int:
        return self._h.vtable[Self]().area(self._h.data)

def main():
    var x: Existential[DynShape].Any[Shape] = 5  # Int is not a Shape
