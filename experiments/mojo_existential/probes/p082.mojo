# Probe: trait-indexed adapter universe resolved at compile time.
from std.builtin.rebind import downcast

comptime Opaque = MutOpaquePointer[MutUntrackedOrigin]

def _trait_eq[a: type_of(AnyType), b: type_of(AnyType)]() -> Bool:
    return __mlir_attr[
        `#kgen.param.identical<`,
        `#kgen.type<`, +a, `> : !kgen.type`, `,`,
        `#kgen.type<`, +b, `> : !kgen.type`,
        `> : !kgen.scalar<bool>`,
    ]

trait TraitAdapter(Movable):
    comptime Trait: type_of(AnyType)
    @staticmethod
    def adapt[T: Self.Trait](data: Opaque) -> Self: ...

trait Greet:
    def greet(self) -> String: ...

trait Shout(Greet):
    def shout(self) -> String: ...

struct DynGreet(Greet, TraitAdapter):
    comptime Trait = Greet
    var data: Opaque
    var _greet: def(Opaque) thin -> String

    def __init__(out self, data: Opaque, greet: def(Opaque) thin -> String):
        self.data = data
        self._greet = greet

    @staticmethod
    def adapt[T: Greet](data: Opaque) -> Self:
        def greet_thunk(p: Opaque) -> String:
            return p.unsafe_bitcast[T]()[].greet()
        return Self(data, greet_thunk)

    def greet(self) -> String:
        return self._greet(self.data)

struct Universe[*Adapters: TraitAdapter]:
    @staticmethod
    def index_of[U: type_of(AnyType)]() -> Int:
        comptime for i in range(Self.Adapters.length):
            comptime if _trait_eq[Self.Adapters[i].Trait, U]():
                return i
        return -1


@fieldwise_init
struct Foo(Shout, Movable):
    var x: Int
    def greet(self) -> String:
        return String("hi from Foo(", self.x, ")")
    def shout(self) -> String:
        return String("HI FROM FOO(", self.x, ")")

comptime U = Universe[DynGreet]

def call_greet[G: Greet](g: G) -> String:
    return g.greet()

# C: comptime-computed index, concrete composition
comptime _idx = U.index_of[Greet]()
comptime W = downcast[U.Adapters[_idx], Greet & TraitAdapter]
def main():
    var f = Foo(3)
    var p = Pointer(to=f).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]()
    var w = rebind_var[W](U.Adapters[0].adapt[Foo](p))
    print(w.greet(), call_greet(w))
