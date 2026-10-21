from prelude import *
# b: index through a separate parametric Int alias
comptime _Idx[Tr: type_of(AnyType)]: Int = U.index_of[Tr]()
comptime Witness[Tr: type_of(AnyType)] = downcast[U.Adapters[_Idx[Tr]], Tr & TraitAdapter]
comptime W = Witness[Greet]
def main():
    var f = Foo(3)
    var p = Pointer(to=f).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]()
    var w = rebind_var[W](U.Adapters[0].adapt[Foo](p))
    print(w.greet(), call_greet(w))
    _ = f^
