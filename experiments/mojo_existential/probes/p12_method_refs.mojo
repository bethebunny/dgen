# Probe: unbound methods as compile-time function values -> generic thunks.
from std.builtin.globals import global_constant

comptime Opaque = MutOpaquePointer[MutUntrackedOrigin]

trait Shape:
    def area(self) -> Int: ...
    def scale(mut self, k: Int): ...

@fieldwise_init
struct Sq(Shape, Movable):
    var s: Int
    def area(self) -> Int: return self.s * self.s
    def scale(mut self, k: Int): self.s *= k

# generic "erase the receiver" helpers, one per arity
def erase0[T: AnyType, R: AnyType, //, f: def(T) thin -> R](p: Opaque) -> R:
    return f(p.unsafe_bitcast[T]()[])

def erase_mut1[T: AnyType, A: AnyType, //, f: def(mut T, A) thin -> None](p: Opaque, a: A):
    f(p.unsafe_bitcast[T]()[], a)

def main():
    var x = Sq(3)
    var p = Pointer(to=x).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]()
    # (a) unbound method as a value
    var g = Sq.area
    print("unbound call:", g(x))
    var t: def(Opaque) thin -> Int = erase0[Sq.area]
    print("erased thunk:", t(p))
    # (b) tuple-of-fn-ptr vtable as a global constant
    # (b) a Tuple of fn pointers is rejected by global_constant: Tuple is not
    #     marked trivially copyable, so vtables must stay explicit structs.
    var sc: def(Opaque, Int) thin -> None = erase_mut1[Sq.scale]
    sc(p, 2)
    print("erased mut thunk:", x.s)
    _ = x^
