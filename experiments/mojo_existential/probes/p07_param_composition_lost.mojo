from prelude import *
# a: parametric alias, then concrete alias of it
comptime Witness[Tr: type_of(AnyType)] = downcast[U.Adapters[U.index_of[Tr]()], Tr & TraitAdapter]
comptime W = Witness[Greet]
def main():
    var f = Foo(3)
    var p = Pointer(to=f).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]()
    var w = rebind_var[W](U.Adapters[0].adapt[Foo](p))
    print(w.greet(), call_greet(w))
    _ = f^
