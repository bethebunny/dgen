from existential import *


# ---- user traits -------------------------------------------------------------
trait MyTrait:
    def method(self) -> String:
        ...


trait MySubTrait(MyTrait):
    def sub_method(self) -> Int:
        ...


trait SomeOtherTrait:
    def other(self) -> String:
        ...


# ---- user types --------------------------------------------------------------
@fieldwise_init
struct Foo(MySubTrait, Movable):
    var x: Int

    def method(self) -> String:
        return String("Foo.method x=", self.x)

    def sub_method(self) -> Int:
        return self.x * 10


@fieldwise_init
struct Bar(MyTrait, SomeOtherTrait, Movable):
    var s: String

    def method(self) -> String:
        return String("Bar.method s=", self.s)

    def other(self) -> String:
        return "Bar.other"


# ---- adapters: one per trait, hand-written (see NOTES.md) --------------------
@fieldwise_init
struct MyTraitVT(Movable):
    var method: def(Opaque) thin -> String


struct DynMyTrait(MyTrait, TraitAdapter):
    comptime Trait = MyTrait
    comptime VTable = MyTraitVT
    var _h: Header

    @staticmethod
    def vtable_for[T: MyTrait]() -> MyTraitVT:
        def method(p: Opaque) -> String:
            return p.unsafe_bitcast[T]()[].method()

        return MyTraitVT(method)

    def method(self) -> String:
        return self._h.vtable[Self]().method(self._h.data)


@fieldwise_init
struct MySubTraitVT(Movable):
    var base: MyTraitVT
    var sub_method: def(Opaque) thin -> Int


struct DynMySubTrait(MySubTrait, TraitAdapter):
    comptime Trait = MySubTrait
    comptime VTable = MySubTraitVT
    var _h: Header

    @staticmethod
    def vtable_for[T: MySubTrait]() -> MySubTraitVT:
        def sub_method(p: Opaque) -> Int:
            return p.unsafe_bitcast[T]()[].sub_method()

        return MySubTraitVT(DynMyTrait.vtable_for[T](), sub_method)

    def method(self) -> String:
        return self._h.vtable[Self]().base.method(self._h.data)

    def sub_method(self) -> Int:
        return self._h.vtable[Self]().sub_method(self._h.data)


@fieldwise_init
struct SomeOtherTraitVT(Movable):
    var other: def(Opaque) thin -> String


struct DynSomeOtherTrait(SomeOtherTrait, TraitAdapter):
    comptime Trait = SomeOtherTrait
    comptime VTable = SomeOtherTraitVT
    var _h: Header

    @staticmethod
    def vtable_for[T: SomeOtherTrait]() -> SomeOtherTraitVT:
        def other(p: Opaque) -> String:
            return p.unsafe_bitcast[T]()[].other()

        return SomeOtherTraitVT(other)

    def other(self) -> String:
        return self._h.vtable[Self]().other(self._h.data)


comptime U = Existential[DynMyTrait, DynMySubTrait, DynSomeOtherTrait]
comptime AnyOf[Base: type_of(AnyType)] = Any[Base, U]


# Generic code that only knows the trait still works on a witness.
def use_generically[T: MySubTrait](t: T) -> String:
    return String(t.method(), " / ", t.sub_method())


def main() raises:
    var foo = Foo(7)
    var ts: List[AnyOf[MyTrait]] = [foo^, Bar("hello")]

    for i in range(len(ts)):
        print(i, ts[i].type_name())

    ref t = trait_cast[MySubTrait](ts[0])
    print(t.method(), t.sub_method())
    print(use_generically(t))

    ref foo2 = cast[Foo](t)
    print("cast[Foo] ->", foo2.x)

    try:
        _ = cast[Int](t)
    except e:
        print("raised:", e)

    try:
        _ = trait_cast[SomeOtherTrait](ts[0])
    except e:
        print("raised:", e)

    ref o = trait_cast[SomeOtherTrait](ts[1])
    print(o.other())
