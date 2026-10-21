# Probe: is a function instantiated on downcast[T, Tr] the same as on T?
from existential import type_id
from std.builtin.rebind import downcast

trait Greet:
    def greet(self) -> String: ...

@fieldwise_init
struct Foo(Greet, Movable):
    def greet(self) -> String: return "foo"
    def self_id(self) -> Int: return type_id[Self]()

def id_via_generic[T: Greet](t: T) -> Int:
    return type_id[T]()

def main():
    print("Foo == downcast[Foo, Greet]:", type_id[Foo]() == type_id[downcast[Foo, Greet]]())
    print("Foo == downcast[Foo, Movable]:", type_id[Foo]() == type_id[downcast[Foo, Movable]]())
    print("name:", reflect[downcast[Foo, Greet]].name())
    ref f = rebind[downcast[Foo, Greet]](Foo())
    print("via generic:", id_via_generic(f) == type_id[Foo]())
    print("Self in method:", f.self_id() == type_id[Foo]())
