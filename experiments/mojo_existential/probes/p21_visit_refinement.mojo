# Probe: what adapter-free `visit` can and can't stand in for.
from std.builtin.rebind import downcast
from existential import *


trait Shape:
    def area(self) -> Int:
        ...


trait Resizable(Shape):
    def scale(mut self, k: Int):
        ...


trait Named:
    def name(self) -> String:
        ...


@fieldwise_init
struct Sq(Movable, Named, Resizable):
    var s: Int

    def area(self) -> Int:
        return self.s * self.s

    def scale(mut self, k: Int):
        self.s *= k

    def name(self) -> String:
        return "sq"


@fieldwise_init
struct Tri(Movable, Shape):
    var b: Int

    def area(self) -> Int:
        return self.b * 3


def _id_of[X: AnyType](ref x: X) -> Int:
    return Pointer(to=x).unsafe_bitcast[Header]()[].type_id()


struct Candidates[*Types: Movable]:
    # read-only visitor: matches ordinary `def g[T: Tr](t: T)` free functions
    @staticmethod
    def visit[
        R: Movable,
        X: AnyType,
        //,
        Tr: type_of(AnyType),
        f: def[T: Tr](T) thin -> R,
    ](x: X) raises -> R:
        comptime for i in range(Self.Types.length):
            comptime T = Self.Types[i]
            comptime if conforms_to(T, Tr):
                if _id_of(x) == type_id[T]():
                    return f(cast[downcast[T, Tr]](x))
        raise Error("no candidate conforms")

    # mutating visitor with one forwarded extra argument
    @staticmethod
    def visit_mut1[
        R: Movable,
        A: AnyType,
        X: AnyType,
        //,
        Tr: type_of(AnyType),
        f: def[T: Tr](mut T, A) thin -> R,
    ](mut x: X, a: A) raises -> R:
        comptime for i in range(Self.Types.length):
            comptime T = Self.Types[i]
            comptime if conforms_to(T, Tr):
                if _id_of(x) == type_id[T]():
                    return f(cast[downcast[T, Tr]](x), a)
        raise Error("no candidate conforms")


# (1) a pre-existing free function expecting a refined trait, passed as-is
def describe[T: Resizable](t: T) -> String:
    return String("resizable with area ", t.area())


# (2) extra argument forwarded through visit
def scale_by[T: Resizable](mut t: T, k: Int) -> Int:
    t.scale(k)
    return t.area()


# (3) "trait_cast" to other traits inside the visitor = compile-time branches
def inspect[T: Shape](t: T) -> String:
    var out = String("area ", t.area())
    comptime if conforms_to(T, Named):
        out += String(", named ", t.name())  # sideways cast, no runtime check
    comptime if conforms_to(T, Resizable):
        out += String(" | ", describe(t))  # pass on to a refined free fn
    return out


comptime known = Candidates[Sq, Tri]
comptime none = Existential[]


def main() raises:
    var a: none.Any[Shape] = Sq(3)
    var b: none.Any[Shape] = Tri(4)
    print("(1)", known.visit[Resizable, describe](a))
    print("(2)", known.visit_mut1[Resizable, scale_by](a, 2))
    print("(3) Sq: ", known.visit[Shape, inspect](a))
    print("(3) Tri:", known.visit[Shape, inspect](b))
    try:
        _ = known.visit[Resizable, describe](b)
    except e:
        print("(1) Tri as Resizable:", e)
