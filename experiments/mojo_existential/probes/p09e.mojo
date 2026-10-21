from prelude import *
# e: parametric trait target only (no composition), unresolved inner
comptime Witness[Tr: type_of(AnyType)] = downcast[U.Adapters[U.index_of[Tr]()], Tr]
comptime W = Witness[Greet]
def main():
    var f = Foo(3)
    var p = Pointer(to=f).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]()
    var a = U.Adapters[0].adapt[Foo](p)
    ref w = rebind[W](a)
    print(w.greet(), call_greet(w))
    _ = f^
