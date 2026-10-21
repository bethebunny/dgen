from prelude import *
# g: generic function returning ref to downcast-parametric type
def as_w[Tr: type_of(AnyType)](ref a: U.Adapters[U.index_of[Tr]()]) -> ref[a] downcast[U.Adapters[U.index_of[Tr]()], Tr]:
    return rebind[downcast[U.Adapters[U.index_of[Tr]()], Tr]](a)
comptime W = Int
def main():
    var f = Foo(3)
    var p = Pointer(to=f).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]()
    var a = U.Adapters[0].adapt[Foo](p)
    ref w = as_w[Greet](a)
    print(w.greet(), call_greet(w))
    _ = f^
