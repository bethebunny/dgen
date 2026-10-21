from std.builtin.rebind import downcast

comptime Opaque = MutOpaquePointer[MutUntrackedOrigin]

def _trait_eq[a: type_of(AnyType), b: type_of(AnyType)]() -> Bool:
    return __mlir_attr[
        `#kgen.param.identical<`,
        `#kgen.type<`, +a, `> : !kgen.type`, `,`,
        `#kgen.type<`, +b, `> : !kgen.type`,
        `> : !kgen.scalar<bool>`,
    ]

trait TraitAdapter(Movable, Deinitable):
    comptime Trait: type_of(AnyType)
    @staticmethod
    def adapt[T: Self.Trait](data: Opaque) -> Self: ...

trait Greet:
    def greet(self) -> String: ...

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
struct Foo(Greet, Movable):
    var x: Int
    def greet(self) -> String:
        return String("hi from Foo(", self.x, ")")

def call_greet[G: Greet](g: G) -> String:
    return g.greet()

comptime U = Universe[DynGreet]
