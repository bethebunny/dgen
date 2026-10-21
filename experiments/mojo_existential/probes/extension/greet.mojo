from existential import *


trait Greet:
    def greet(self) -> String:
        ...


@fieldwise_init
struct GreetVT(ImplicitlyCopyable):
    var greet: def(Opaque) thin -> String


struct DynGreet(Greet, TraitAdapter):
    comptime Trait = Greet
    comptime VTable = GreetVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Greet]() -> GreetVT:
        return GreetVT(thunk[T.greet])

    def greet(self) -> String:
        return self._h.vtable[Self]().greet(self._h.data)


def generic_conforms[T: AnyType]() -> Bool:
    # evaluated in a module that never imports the extension
    return conforms_to(T, Greet)
