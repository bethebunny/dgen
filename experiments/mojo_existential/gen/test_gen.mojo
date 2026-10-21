# Uses only *generated* adapters (dyn_shapes.mojo); no hand-written ones.
from std.testing import assert_equal, assert_raises
from existential import *
from shapes import Shape, Resizable
from dyn_shapes import DynShape, DynResizable


@fieldwise_init
struct Box(Movable, Resizable):
    var side: Int
    var label: String

    def area(self) -> Int:
        return self.side * self.side

    def name(self) -> String:
        return String("box ", self.label)

    def scale(mut self, k: Int):
        self.side *= k

    def set_label(mut self, var label: String):
        self.label = label^

    def checked_div(self, d: Int) raises -> Int:
        if d == 0:
            raise Error("divide by zero")
        return self.side // d


@fieldwise_init
struct Dot(Movable, Shape):
    def area(self) -> Int:
        return 0

    def name(self) -> String:
        return "dot"


comptime dyn = Existential[DynShape, DynResizable]
comptime trait_cast = dyn.trait_cast


def total_area[S: Shape](s: S) -> Int:
    return s.area()


def main() raises:
    var xs: List[dyn.Any[Shape]] = [Box(3, "a"), Dot()]
    assert_equal(xs[0][].name(), "box a")
    ref r = trait_cast[Resizable](xs[0])
    r.scale(2)  # mut self
    r.set_label(String("b"))  # var arg
    assert_equal(r.area(), 36)  # inherited method through the subtrait
    assert_equal(r.name(), "box b")
    assert_equal(r.checked_div(4), 1)
    with assert_raises(contains="divide by zero"):
        _ = r.checked_div(0)  # raises propagates through the vtable
    assert_equal(total_area(r), 36)  # generated adapter conforms to the trait
    with assert_raises(contains="does not conform"):
        _ = trait_cast[Resizable](xs[1])
    print("generated adapters: all checks passed")
