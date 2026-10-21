# Probe: thin function pointers in a struct, built generically from T: Trait.
trait Greet:
    def greet(self) -> String: ...

@fieldwise_init
struct Foo(Greet, Movable):
    var x: Int
    def greet(self) -> String:
        return String("Foo(", self.x, ")")

struct Vt(Copyable):
    var greet: def(ImmPointer[NoneType, _]) thin -> String

    def __init__[T: Greet](out self, *, for_type: type_of(T)):
        def greet_thunk(p: ImmPointer[NoneType, _]) -> String:
            return p.unsafe_bitcast[T]()[].greet()
        self.greet = greet_thunk

def main():
    var f = Foo(7)
    var vt = Vt(for_type=Foo)
    var p = Pointer(to=f).unsafe_bitcast[NoneType]()
    print(vt.greet(p))
