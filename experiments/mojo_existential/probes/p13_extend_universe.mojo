# Probe: can traits be added to a universe after it is defined?
from existential import *


# ---- "library": knows only Shape --------------------------------------------
trait Shape:
    def area(self) -> Int:
        ...


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


comptime base = Existential[DynShape]


@fieldwise_init
struct Rect(Movable, Shape):
    var w: Int
    var h: Int

    def area(self) -> Int:
        return self.w * self.h

    def perimeter(self) -> Int:
        return 2 * (self.w + self.h)


# (b) library code generic over the universe it boxes into
def make_rect[*Ad: TraitAdapter]() -> AnyBox[Shape, *Ad]:
    return Rect(2, 3)


# ---- "application": defines a new trait after the fact ------------------------
trait Perimeter:
    def perimeter(self) -> Int:
        ...


# Rect didn't declare Perimeter; conformance must be declared on the type, so
# use a wrapper-free route: a type that does declare it.
@fieldwise_init
struct Square(Movable, Perimeter, Shape):
    var s: Int

    def area(self) -> Int:
        return self.s * self.s

    def perimeter(self) -> Int:
        return 4 * self.s


def make_square[*Ad: TraitAdapter]() -> AnyBox[Shape, *Ad]:
    return Square(5)


@fieldwise_init
struct PerimeterVT(ImplicitlyCopyable):
    var perimeter: def(Opaque) thin -> Int


struct DynPerimeter(Perimeter, TraitAdapter):
    comptime Trait = Perimeter
    comptime VTable = PerimeterVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Perimeter]() -> PerimeterVT:
        return PerimeterVT(thunk[T.perimeter])

    def perimeter(self) -> Int:
        return self._h.vtable[Self]().perimeter(self._h.data)


# (a) extend the library's universe by concatenation
comptime app = base.Extend[DynPerimeter]


def main() raises:
    # (a)+(b): boxed inside "library" code, under the app's extended universe
    var s = make_square[*app.Adapters]()
    print(
        "(a+b) extended universe:",
        app.trait_cast[Perimeter](s).perimeter(),
        "/ area",
        s[].area(),
    )

    # (c) boxed under the *old* universe: the new trait is invisible...
    var old = make_square[*base.Adapters]()
    try:
        _ = trait_cast_via[DynPerimeter](old)
    except e:
        print("(c) old universe:", e)
    # ...until this (type, trait) pair is registered at runtime
    register[Square, DynPerimeter]()
    print("(c) after register:", trait_cast_via[DynPerimeter](old).perimeter())

    # NB: base.Any[Shape] and app.Any[Shape] are distinct, non-convertible types
    # (reflect prints both as AnyBox[<unprintable>, ...]).
