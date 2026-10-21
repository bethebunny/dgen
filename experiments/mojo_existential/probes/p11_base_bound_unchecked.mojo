# Probe: `T: Self.Base & Movable & Deinitable` with a trait-typed *parameter*
# Base. Mojo 1.1 drops `Self.Base` from the composition (the only hint is a
# misleading "redundant trait composition: 'Movable' already implies
# 'AnyType'" warning), so the bound is NOT enforced.
#
# STATUS: bug. This program compiles and runs, storing an Int in a Box[Greet].
# Workaround: `def __init__[T: Movable & Deinitable](...) where
# conforms_to(T, Self.Base)`, which is enforced (see
# ../compile_fail/base_bound.mojo).
trait Greet:
    def greet(self) -> String: ...

struct Box[Base: type_of(AnyType)]:
    var n: Int
    @implicit
    def __init__[T: Self.Base & Movable & Deinitable](out self, var value: T):
        self.n = 1

def main():
    var a: Box[Greet] = 5      # Int is not Greet: should be rejected
    print("accepted an Int into Box[Greet]:", a.n)
