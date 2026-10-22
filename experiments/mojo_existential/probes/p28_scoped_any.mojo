# Mitigation: an origin-carrying ScopedAny (like dyn Trait + 'a). Keeps the
# anchor alive (ASan clean) and can't escape its scope -- but the anchor is
# chosen by the caller and not checked against what T actually borrows.
# Uncomment escape() to see: cannot implicitly convert 'ScopedAny[origin_of(ys)]'.
# Probe: Any has no origin parameter, so boxing a value that *borrows* erases
# the borrow. Rust rejects the equivalent (`dyn Trait + 'a`, `dyn Any: 'static`).
# STATUS: compiles; ASan reports heap-use-after-free.
from existential import *


trait Peek:
    def peek(self) -> Int:
        ...


@fieldwise_init
struct PeekVT(ImplicitlyCopyable):
    var peek: def(Opaque) thin -> Int


struct DynPeek(Peek, TraitAdapter):
    comptime Trait = Peek
    comptime VTable = PeekVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Peek]() -> PeekVT:
        return PeekVT(thunk[T.peek])

    def peek(self) -> Int:
        return self._h.vtable[Self]().peek(self._h.data)


@fieldwise_init
struct Borrower[o: ImmOrigin](Movable, Peek):
    var p: ImmPointer[Int, Self.o]

    def peek(self) -> Int:
        return self.p[]


# Mitigation: a scoped Any that carries the borrowed origin, like `dyn Peek + 'a`.
struct ScopedAny[o: ImmOrigin](Movable):
    var any: Existential[DynPeek].Any[Peek]
    var _anchor: ImmPointer[
        NoneType, Self.o
    ]  # phantom: makes the borrow visible

    def __init__[
        T: Peek, A: AnyType
    ](out self, var value: T, ref[Self.o] anchor: A) where conforms_to(
        T, Movable & Deinitable
    ):
        self.any = value^
        self._anchor = Pointer(to=anchor).unsafe_bitcast[NoneType]()


def main():
    var xs: List[Int] = [1, 2, 3]
    var s = ScopedAny(Borrower(Pointer(to=xs[1])), anchor=xs)
    print(s.any[].peek())


# def escape() -> ScopedAny[ImmUnsafeAnyOrigin]:
#     var ys: List[Int] = [7]
#     return ScopedAny(Borrower(Pointer(to=ys[0])), anchor=ys)
