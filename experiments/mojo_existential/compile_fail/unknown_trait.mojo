# EXPECT-ERROR: trait has no adapter in this universe
from existential import *

trait Shape:
    def area(self) -> Int: ...

trait Unlisted:
    def f(self): ...

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

@fieldwise_init
struct Sq(Shape, Movable):
    var s: Int
    def area(self) -> Int: return self.s

def main() raises:
    var x: Existential[DynShape].Any[Shape] = Sq(1)
    _ = Existential[DynShape].trait_cast[Unlisted](x)
