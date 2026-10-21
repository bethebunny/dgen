# Probe: does an `__extension` conformance work with the existential machinery?
from existential import *
from greet import Greet, DynGreet, GreetVT, generic_conforms
from lib import Point


__extension Point(Greet):
    def greet(self) -> String:
        return String("hi from ", self.x, ",", self.y)


comptime dyn = Existential[DynGreet]


def generic_bound[G: Greet](g: G) -> String:
    return g.greet()


def main() raises:
    print("direct generic call:", generic_bound(Point(1, 2)))
    print("conforms_to here:", conforms_to(Point, Greet))
    print("conforms_to in other module:", generic_conforms[Point]())
    comptime if conforms_to(Point, Greet):
        print("comptime if: True")
    else:
        print("comptime if: False")
    # vtable built directly from the extension method
    comptime vt = DynGreet.vtable_for[Point]()
    var p = Point(5, 6)
    print(
        "vtable_for[Point]:",
        vt.greet(
            Pointer(to=p)
            .unsafe_bitcast[NoneType]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        ),
    )
    _ = p^
