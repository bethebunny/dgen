# Probe: Any has no origin parameter, so boxing a value that *borrows* erases
# the borrow. Rust rejects the equivalent (`dyn Trait + 'a`, `dyn Any: 'static`).
# STATUS: compiles; ASan reports heap-use-after-free.
from existential import *

trait Peek:
    def peek(self) -> Int: ...

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

def make() -> Existential[DynPeek].Any[Peek]:
    var xs: List[Int] = [1, 2, 3]
    return Borrower(Pointer(to=xs[1]))  # xs dies at end of scope

def main():
    var a = make()
    print(a[].peek())
