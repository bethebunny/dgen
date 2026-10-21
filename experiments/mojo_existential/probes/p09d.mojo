from prelude import *
# d: parametric, but index computed from Tr via a struct-level comptime member
struct Lookup[Tr: type_of(AnyType)]:
    comptime idx = U.index_of[Self.Tr]()
    comptime W = downcast[U.Adapters[Self.idx], Self.Tr & TraitAdapter]
comptime W = Lookup[Greet].W
def main():
    var f = Foo(3)
    var p = Pointer(to=f).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]()
    var w = rebind_var[W](U.Adapters[0].adapt[Foo](p))
    print(w.greet(), call_greet(w))
    _ = f^
