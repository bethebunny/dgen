# Probe: struct parameterised by a trait; method bounds use that parameter;
# comptime-if conforms_to refinement with a *parameter* trait.
trait Greet:
    def greet(self) -> String: ...

@fieldwise_init
struct Foo(Greet, Movable):
    var x: Int
    def greet(self) -> String:
        return String("Foo(", self.x, ")")

struct Box[Base: type_of(AnyType)]:
    var n: Int
    @implicit
    def __init__[T: Self.Base & Movable](out self, var value: T):
        self.n = 1

def try_greet[U: type_of(AnyType), T: AnyType](x: T) -> String:
    comptime if conforms_to(T, Greet):
        return x.greet()
    else:
        return "nope"

trait HasTrait:
    comptime Trait: type_of(AnyType)

struct GreetTag(HasTrait):
    comptime Trait = Greet

def via_member[D: HasTrait, T: AnyType](x: T) -> Bool:
    return conforms_to(T, D.Trait)

def main():
    var b: Box[Greet] = Foo(1)
    print(b.n)
    print(try_greet[Greet](Foo(2)), try_greet[Greet](3))
    print(via_member[GreetTag](Foo(3)), via_member[GreetTag](4))
