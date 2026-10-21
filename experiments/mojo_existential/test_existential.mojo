from std.ffi import _Global
from std.os import abort
from std.sys import size_of
from std.testing import assert_equal, assert_true, assert_false, assert_raises

from existential import *


# ---- traits ------------------------------------------------------------------
trait Shape:
    def area(self) -> Int:
        ...


trait Named:
    def name(self) -> String:
        ...


trait Counter:
    def bump(mut self):
        ...

    def count(self) -> Int:
        ...


# ---- adapters ----------------------------------------------------------------
@fieldwise_init
struct ShapeVT(ImplicitlyCopyable, Movable):
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


@fieldwise_init
struct NamedVT(ImplicitlyCopyable, Movable):
    var name: def(Opaque) thin -> String


struct DynNamed(Named, TraitAdapter):
    comptime Trait = Named
    comptime VTable = NamedVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Named]() -> NamedVT:
        return NamedVT(thunk[T.name])

    def name(self) -> String:
        return self._h.vtable[Self]().name(self._h.data)


@fieldwise_init
struct CounterVT(ImplicitlyCopyable, Movable):
    var bump: def(Opaque) thin -> None
    var count: def(Opaque) thin -> Int


struct DynCounter(Counter, TraitAdapter):
    comptime Trait = Counter
    comptime VTable = CounterVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Counter]() -> CounterVT:
        return CounterVT(thunk_mut[T.bump], thunk[T.count])

    def bump(mut self):
        self._h.vtable[Self]().bump(self._h.data)

    def count(self) -> Int:
        return self._h.vtable[Self]().count(self._h.data)


comptime dyn = Existential[DynShape, DynNamed, DynCounter]
comptime Any = dyn.Any
comptime trait_cast = dyn.trait_cast


# ---- concrete types, with live-instance accounting ---------------------------
def _zero() -> Int:
    return 0


# A named process-global (runtime-backed): the only mutable global storage
# reachable from Mojo code today.
comptime _live = _Global["existential_test_live", _zero]


def live() -> Pointer[Int, MutUntrackedOrigin]:
    try:
        return _live.get_or_create_ptr()
    except:
        abort("no global")


struct Square(Copyable, Named, Shape):
    var side: Int

    def __init__(out self, side: Int):
        self.side = side
        live()[] += 1

    def __init__(out self, *, copy: Self):
        self.side = copy.side
        live()[] += 1

    def __init__(out self, *, deinit move: Self):
        self.side = move.side

    def __deinit__(deinit self):
        live()[] -= 1

    def area(self) -> Int:
        return self.side * self.side

    def name(self) -> String:
        return "square"


struct Tally(Counter, Movable, Named):
    var n: Int

    def __init__(out self):
        self.n = 0
        live()[] += 1

    def __init__(out self, *, deinit move: Self):
        self.n = move.n

    def __deinit__(deinit self):
        live()[] -= 1

    def bump(mut self):
        self.n += 1

    def count(self) -> Int:
        return self.n

    def name(self) -> String:
        return "tally"


# ---- tests -------------------------------------------------------------------
def test_layout() raises:
    assert_equal(size_of[Any[Shape]](), 16)
    assert_equal(size_of[dyn.Witness[Shape]](), 16)


def test_base_view_and_dispatch() raises:
    var before = live()[]
    var s: Any[Shape] = Square(3)
    assert_equal(live()[], before + 1)
    assert_equal(s[].area(), 9)
    assert_equal(s.type_name(), "test_existential.Square")


def test_trait_cast_sideways() raises:
    var s: Any[Shape] = Square(4)
    ref n = trait_cast[Named](s)
    assert_equal(n.name(), "square")
    # witness -> witness -> concrete
    ref sh = trait_cast[Shape](n)
    assert_equal(sh.area(), 16)
    assert_equal(cast[Square](sh).side, 4)


def test_failures_raise() raises:
    var s: Any[Shape] = Square(2)
    with assert_raises(contains="does not conform"):
        _ = trait_cast[Counter](s)
    with assert_raises(contains="cannot cast"):
        _ = cast[Tally](s)
    with assert_raises(contains="cannot cast"):
        _ = cast[Int](trait_cast[Named](s))


def test_mutation_through_witness_and_cast() raises:
    var c: Any[Named] = Tally()
    ref w = trait_cast[Counter](c)
    w.bump()
    w.bump()
    assert_equal(w.count(), 2)
    cast[Tally](c).n += 10
    assert_equal(trait_cast[Counter](c).count(), 12)


def test_meta_is_singleton() raises:
    var a: Any[Shape] = Square(1)
    var b: Any[Shape] = Square(2)
    var ha = Pointer(to=a).unsafe_bitcast[Header]()[]
    var hb = Pointer(to=b).unsafe_bitcast[Header]()[]
    assert_equal(Int(ha.meta), Int(hb.meta))
    assert_true(Int(ha.data) != Int(hb.data))
    # and the per-(T, trait) vtable is a singleton too
    assert_equal(ha.witness[DynNamed](), hb.witness[DynNamed]())


def test_list_heterogeneous() raises:
    var xs: List[Any[Named]] = [Square(5), Tally(), Square(6)]
    var areas = 0
    var non_shapes = 0
    for i in range(len(xs)):
        try:
            areas += trait_cast[Shape](xs[i]).area()
        except:
            non_shapes += 1
    assert_equal(areas, 25 + 36)
    assert_equal(non_shapes, 1)


def test_clone() raises:
    var s: Any[Shape] = Square(7)
    var t = dyn.clone[Named](s)
    cast[Square](t).side = 1
    assert_equal(cast[Square](s).side, 7)
    assert_equal(t[].name(), "square")
    var c: Any[Named] = Tally()
    with assert_raises(contains="not Copyable"):
        _ = dyn.clone[Named](c)
    with assert_raises(contains="does not conform"):
        _ = dyn.clone[Shape](c)


def generic_area[S: Shape](s: S) -> Int:
    return s.area()


def test_witness_in_generic_code() raises:
    var s: Any[Named] = Square(8)
    assert_equal(generic_area(trait_cast[Shape](s)), 64)


# A trait *outside* the universe, plus an adapter for it.
trait Perimeter:
    def perimeter(self) -> Int:
        ...


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


@fieldwise_init
struct Rect(Movable, Named, Perimeter):
    var w: Int
    var h: Int

    def name(self) -> String:
        return "rect"

    def perimeter(self) -> Int:
        return 2 * (self.w + self.h)


def test_open_world_registry() raises:
    var r: Any[Named] = Rect(2, 3)
    # Universe-free cast to a universe trait works without registration.
    assert_equal(trait_cast_via[DynNamed](r).name(), "rect")
    # Perimeter isn't in the universe: unknown until registered.
    with assert_raises(contains="does not conform"):
        _ = trait_cast_via[DynPerimeter](r)
    register[Rect, DynPerimeter]()
    assert_equal(trait_cast_via[DynPerimeter](r).perimeter(), 10)


def perimeter_of_square(sq: Square) -> Int:
    return 4 * sq.side


def test_retroactive_conformance() raises:
    # Square never declared Perimeter; supply its witness table instead.
    var s: Any[Shape] = Square(3)
    with assert_raises(contains="does not conform"):
        _ = trait_cast_via[DynPerimeter](s)
    unsafe_register_as[
        Square, DynPerimeter, PerimeterVT(thunk[perimeter_of_square])
    ]()
    assert_equal(trait_cast_via[DynPerimeter](s).perimeter(), 12)


comptime small = Existential[DynShape]
comptime big = small.Extend[DynNamed, DynCounter]


def box_shape[*Ad: TraitAdapter](var s: Square) -> AnyBox[Shape, *Ad]:
    # Library-style code, generic over whichever universe the caller uses.
    return s^


def test_extend_universe() raises:
    var b = box_shape[*big.Adapters](Square(2))
    assert_equal(big.trait_cast[Named](b).name(), "square")
    assert_equal(big.index_of[Shape](), 0)
    assert_equal(big.index_of[Counter](), 2)
    # A value boxed under the smaller universe doesn't learn the new traits.
    var s = box_shape[*small.Adapters](Square(2))
    with assert_raises(contains="does not conform"):
        _ = trait_cast_via[DynNamed](s)


def main() raises:
    var before = live()[]
    test_layout()
    test_base_view_and_dispatch()
    test_trait_cast_sideways()
    test_failures_raise()
    test_mutation_through_witness_and_cast()
    test_meta_is_singleton()
    test_list_heterogeneous()
    test_clone()
    test_witness_in_generic_code()
    test_open_world_registry()
    test_retroactive_conformance()
    test_extend_universe()
    assert_equal(live()[], before, "leaked or double-freed instances")
    print("all tests passed; live instances:", live()[])
