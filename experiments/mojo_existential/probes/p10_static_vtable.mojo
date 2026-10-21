# Probe: fn pointers inside a comptime struct value placed in rodata via
# global_constant; and aliasing a static method as a free function.
from std.builtin.globals import global_constant

trait Greet:
    def greet(self) -> String: ...

@fieldwise_init
struct Foo(Greet, Movable):
    var x: Int
    def greet(self) -> String:
        return String("Foo(", self.x, ")")

@fieldwise_init
struct VT(ImplicitlyCopyable, Movable):
    var greet: def(ImmOpaquePointer[_]) thin -> String

def _greet_thunk[T: Greet](p: ImmOpaquePointer[_]) -> String:
    return p.unsafe_bitcast[T]()[].greet()

def vtable_for[T: Greet]() -> VT:
    return VT(_greet_thunk[T])

def static_vt[T: Greet]() -> ImmPointer[VT, ImmStaticOrigin]:
    comptime vt = vtable_for[T]()
    return Pointer(to=global_constant[vt]())

struct NS:
    @staticmethod
    def hello[T: AnyType](x: Int) -> String:
        return String(reflect[T].name(), " ", x)

comptime hello = NS.hello

def main():
    var f = Foo(5)
    var a = static_vt[Foo]()
    var b = static_vt[Foo]()
    print(a[].greet(Pointer(to=f).unsafe_bitcast[NoneType]()))
    print("same address:", Int(a) == Int(b))
    print(hello[Foo](3))
    _ = f^
