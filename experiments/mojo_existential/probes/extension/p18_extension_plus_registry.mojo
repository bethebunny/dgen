# Probe: method-only `__extension` + unsafe_register_as = retroactive conformance
# that the existential machinery can actually use.
from existential import *
from greet import Greet, DynGreet, GreetVT
from lib import Point


__extension Point:  # adds the method; conformance via extension is broken (p17)
    def greet(self) -> String:
        return String("hi from ", self.x, ",", self.y)


comptime dyn = Existential[DynGreet]


def main() raises:
    # Box under some base trait Point *does* declare: use Writable via AnyType?
    # Point declares only Copyable/Writable, so box it in an Any[Copyable].
    var a: Existential[DynGreet].Any[Copyable] = Point(1, 2)
    unsafe_register_as[Point, DynGreet, GreetVT(thunk[Point.greet])]()
    print(dyn.trait_cast[Greet](a).greet())
