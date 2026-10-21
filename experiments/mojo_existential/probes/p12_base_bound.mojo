# Probe: is `T: Self.Base & Movable & Deinitable` actually enforcing Base?
trait Greet:
    def greet(self) -> String: ...

struct Box[Base: type_of(AnyType)]:
    var n: Int
    @implicit
    def __init__[T: Self.Base & Movable & Deinitable](out self, var value: T):
        self.n = 1

struct Box2[Base: type_of(AnyType)]:
    var n: Int
    @implicit
    def __init__[T: Movable & Deinitable](out self, var value: T) where conforms_to(T, Self.Base):
        self.n = 2

def main():
    var a: Box[Greet] = 5      # Int is not Greet: should be rejected
    var b: Box2[Greet] = 5
    print(a.n, b.n)
