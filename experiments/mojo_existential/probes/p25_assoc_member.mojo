# Probe: an adapter for a trait with an associated comptime member.
from std.builtin.rebind import downcast
from existential import *


trait Container:
    comptime Element: Copyable & Deinitable & Writable

    def first(self) -> Self.Element:
        ...


@fieldwise_init
struct ContainerIntVT(ImplicitlyCopyable):
    var first: def(Opaque) thin -> Int


# One adapter per fixed Element, like Rust's `dyn Container<Element = Int>`.
struct DynContainerInt(Container, TraitAdapter):
    comptime Trait = Container
    comptime VTable = ContainerIntVT
    comptime Element = Int
    var _h: Header

    @staticmethod
    def accepts[T: AnyType]() -> Bool:
        comptime if conforms_to(T, Container):
            return type_eq[downcast[T, Container].Element, Int]()
        return False

    @staticmethod
    def vtable_for[T: Container]() -> ContainerIntVT:
        def first(p: Opaque) -> Int:
            return rebind[Int](p.unsafe_bitcast[T]()[].first())

        return ContainerIntVT(first)

    def first(self) -> Int:
        return self._h.vtable[Self]().first(self._h.data)


@fieldwise_init
struct Ints(Container, Movable):
    comptime Element = Int
    var v: Int

    def first(self) -> Int:
        return self.v


@fieldwise_init
struct Strs(Container, Movable):
    comptime Element = String
    var s: String

    def first(self) -> String:
        return self.s


comptime dyn = Existential[DynContainerInt]


def main() raises:
    var a: dyn.Any[Container] = Ints(3)
    print("Ints first:", dyn.trait_cast[Container](a).first())
    var b: dyn.Any[Container] = Strs(
        "x"
    )  # Element = String: DynContainerInt rejects it
    try:
        _ = dyn.trait_cast[Container](b)
    except e:
        print("Strs:", e)
