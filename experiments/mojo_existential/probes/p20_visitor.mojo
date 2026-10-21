# Probe: adapter-free trait dispatch. Instead of vtables, the *caller* names
# candidate concrete types, and a generic visitor `f[T: Tr]` (a higher-rank
# compile-time function parameter) is instantiated for each candidate that
# conforms to Tr. Dispatch = compare type ids; calls are static (inlinable).
from existential import *
from std.builtin.rebind import downcast


trait Shape:
    def area(self) -> Int:
        ...


trait Resizable(Shape):
    def scale(mut self, k: Int):
        ...


@fieldwise_init
struct Sq(Movable, Resizable):
    var s: Int

    def area(self) -> Int:
        return self.s * self.s

    def scale(mut self, k: Int):
        self.s *= k


@fieldwise_init
struct Tri(Movable, Shape):
    var b: Int

    def area(self) -> Int:
        return self.b * 3


struct Candidates[*Types: Movable]:
    @staticmethod
    def visit[
        R: Movable,
        X: AnyType,
        //,
        Tr: type_of(AnyType),
        f: def[T: Tr](mut T) thin -> R,
    ](mut x: X) raises -> R:
        """Run f[T] on the erased value if it is one of Types and T: Tr."""
        comptime for i in range(Self.Types.length):
            comptime T = Self.Types[i]
            comptime if conforms_to(T, Tr):
                if (
                    Pointer(to=x).unsafe_bitcast[Header]()[].type_id()
                    == type_id[T]()
                ):
                    return f(cast[downcast[T, Tr]](x))
        raise Error("no candidate type conforming to the trait")


def grow_and_measure[T: Resizable](mut s: T) -> Int:
    s.scale(2)
    return s.area()


def measure[T: Shape](mut s: T) -> Int:
    return s.area()


comptime known = Candidates[Sq, Tri]
comptime none = Existential[]  # an Any with an empty adapter universe


def main() raises:
    var a: none.Any[Shape] = Sq(3)
    var b: none.Any[Shape] = Tri(4)
    print("visit Resizable on Sq:", known.visit[Resizable, grow_and_measure](a))
    print("visit Shape on Tri:", known.visit[Shape, measure](b))
    try:
        _ = known.visit[Resizable, grow_and_measure](b)
    except e:
        print("visit Resizable on Tri:", e)
    print("state persisted:", cast[Sq](a).s)
