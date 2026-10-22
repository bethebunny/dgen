# Probe: hand-written adapters for trait features the generator skips.
from existential import *


# ---- a type-erased Writer, so type-parametric methods can go through a vtable
struct ErasedWriter(Writer):
    var _w: Opaque
    var _write: def(Opaque, ImmStringSlice[_]) thin

    def __init__[W: Writer](out self, mut w: W):
        def write(p: Opaque, s: ImmStringSlice[_]):
            p.unsafe_bitcast[W]()[].write_string(s)

        self._w = (
            Pointer(to=w)
            .unsafe_bitcast[NoneType]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        self._write = write

    def write_string(mut self, string: StringSlice):
        self._write(self._w, string)


@fieldwise_init
struct MyErr(Movable, Writable):
    var code: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("MyErr(", self.code, ")")


# ---- the hard traits -----------------------------------------------------------
trait Show:  # (1) type-parametric method
    def show[W: Writer](self, mut w: W):
        ...


trait Labeled(
    Writable
):  # (2) parent trait from the stdlib (write_to is parametric)
    def tag(self) -> Int:
        ...


trait Same:  # (3) Self argument
    def same(self, other: Self) -> Bool:
        ...


trait Cell:  # (4) ref return
    def get(ref self) -> ref[self] Int:
        ...


trait Over:  # (5) overloads
    def f(self, x: Int) -> Int:
        ...

    def f(self, x: String) -> Int:
        ...


trait Risky:  # (6) typed raises
    def risky(self) raises MyErr -> Int:
        ...


# ---- adapters -------------------------------------------------------------------
@fieldwise_init
struct ShowVT(ImplicitlyCopyable):
    var show: def(Opaque, mut ErasedWriter) thin


struct DynShow(Show, TraitAdapter):
    comptime Trait = Show
    comptime VTable = ShowVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Show]() -> ShowVT:
        def show(p: Opaque, mut w: ErasedWriter):
            p.unsafe_bitcast[T]()[].show(w)  # T.show[ErasedWriter]

        return ShowVT(show)

    def show[W: Writer](self, mut w: W):
        var ew = ErasedWriter(w)
        self._h.vtable[Self]().show(self._h.data, ew)


@fieldwise_init
struct LabeledVT(ImplicitlyCopyable):
    var tag: def(Opaque) thin -> Int
    var write_to: def(Opaque, mut ErasedWriter) thin


struct DynLabeled(Labeled, TraitAdapter):
    comptime Trait = Labeled
    comptime VTable = LabeledVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Labeled]() -> LabeledVT:
        def write_to(p: Opaque, mut w: ErasedWriter):
            p.unsafe_bitcast[T]()[].write_to(w)

        return LabeledVT(thunk[T.tag], write_to)

    def tag(self) -> Int:
        return self._h.vtable[Self]().tag(self._h.data)

    def write_to(self, mut writer: Some[Writer]):
        var ew = ErasedWriter(writer)
        self._h.vtable[Self]().write_to(self._h.data, ew)


@fieldwise_init
struct SameVT(ImplicitlyCopyable):
    var same: def(Opaque, Opaque) thin -> Bool


struct DynSame(Same, TraitAdapter):
    comptime Trait = Same
    comptime VTable = SameVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Same]() -> SameVT:
        def same(p: Opaque, q: Opaque) -> Bool:
            return p.unsafe_bitcast[T]()[].same(q.unsafe_bitcast[T]()[])

        return SameVT(same)

    def same(self, other: Self) -> Bool:
        # Runtime check, like Swift's AnyHashable: different erased types are
        # never "the same", and the vtable is only valid for matching types.
        if self._h.type_id() != other._h.type_id():
            return False
        return self._h.vtable[Self]().same(self._h.data, other._h.data)


@fieldwise_init
struct CellVT(ImplicitlyCopyable):
    var get: def(Opaque) thin -> Int  # returns an address


struct DynCell(Cell, TraitAdapter):
    comptime Trait = Cell
    comptime VTable = CellVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Cell]() -> CellVT:
        def get(p: Opaque) -> Int:
            return Int(Pointer(to=p.unsafe_bitcast[T]()[].get()))

        return CellVT(get)

    def get(ref self) -> ref[self] Int:
        var addr = self._h.vtable[Self]().get(self._h.data)
        return (
            Pointer[Int, MutUntrackedOrigin](unsafe_from_address=addr)
            .unsafe_mut_cast[origin_of(self).mut]()
            .unsafe_origin_cast[origin_of(self)]()[]
        )


@fieldwise_init
struct OverVT(ImplicitlyCopyable):
    var f_int: def(Opaque, Int) thin -> Int
    var f_str: def(Opaque, String) thin -> Int


struct DynOver(Over, TraitAdapter):
    comptime Trait = Over
    comptime VTable = OverVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Over]() -> OverVT:
        def f_int(p: Opaque, x: Int) -> Int:
            return p.unsafe_bitcast[T]()[].f(x)

        def f_str(p: Opaque, x: String) -> Int:
            return p.unsafe_bitcast[T]()[].f(x)

        return OverVT(f_int, f_str)

    def f(self, x: Int) -> Int:
        return self._h.vtable[Self]().f_int(self._h.data, x)

    def f(self, x: String) -> Int:
        return self._h.vtable[Self]().f_str(self._h.data, x)


@fieldwise_init
struct RiskyVT(ImplicitlyCopyable):
    var risky: def(Opaque) thin raises MyErr -> Int


struct DynRisky(Risky, TraitAdapter):
    comptime Trait = Risky
    comptime VTable = RiskyVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Risky]() -> RiskyVT:
        def risky(p: Opaque) raises MyErr -> Int:
            return p.unsafe_bitcast[T]()[].risky()

        return RiskyVT(risky)

    def risky(self) raises MyErr -> Int:
        return self._h.vtable[Self]().risky(self._h.data)


# ---- one concrete type implementing everything ----------------------------------
@fieldwise_init
struct Thing(Cell, Labeled, Movable, Over, Risky, Same, Show):
    var n: Int

    def show[W: Writer](self, mut w: W):
        w.write("Thing#", self.n)

    def tag(self) -> Int:
        return self.n * 10

    def write_to(self, mut writer: Some[Writer]):
        writer.write("<thing ", self.n, ">")

    def same(self, other: Self) -> Bool:
        return self.n == other.n

    def get(ref self) -> ref[self] Int:
        # A trait can't say ref[self.n], and Mojo 1.1 won't widen a field's
        # origin to `self`, so even a direct implementation needs this cast.
        return Pointer(to=self.n).unsafe_origin_cast[origin_of(self)]()[]

    def f(self, x: Int) -> Int:
        return self.n + x

    def f(self, x: String) -> Int:
        return self.n + x.byte_length()

    def risky(self) raises MyErr -> Int:
        raise MyErr(self.n)


@fieldwise_init
struct Other(Movable, Same):
    def same(self, other: Self) -> Bool:
        return True


comptime dyn = Existential[
    DynShow, DynLabeled, DynSame, DynCell, DynOver, DynRisky
]


def show_generic[S: Show](s: S) -> String:
    var out = String()
    s.show(out)
    return out


def main() raises:
    var a: dyn.Any[Show] = Thing(4)
    var b: dyn.Any[Same] = Thing(4)
    var c: dyn.Any[Same] = Thing(5)
    var o: dyn.Any[Same] = Other()

    print("(1) parametric show:", show_generic(a[]))
    ref l = dyn.trait_cast[Labeled](a)
    print("(2) Writable parent:", l.tag(), String(l))
    print(
        "(3) Self arg: 4==4",
        dyn.trait_cast[Same](a).same(b[]),
        " 4==5",
        b[].same(c[]),
        " Thing vs Other",
        b[].same(o[]),
    )
    ref cell = dyn.trait_cast[Cell](a)
    cell.get() += 100
    print("(4) ref return, mutated through it:", cast[Thing](a).n)
    ref ov = dyn.trait_cast[Over](a)
    print("(5) overloads:", ov.f(1), ov.f(String("abc")))
    try:
        _ = dyn.trait_cast[Risky](a).risky()
    except e:
        print("(6) typed raises: caught", e)
