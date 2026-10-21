from prelude import *
# c: concrete trait (control, should work)
comptime W = downcast[U.Adapters[U.index_of[Greet]()], Greet & TraitAdapter]
def main():
    var f = Foo(3)
    var p = Pointer(to=f).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]()
    var w = rebind_var[W](U.Adapters[0].adapt[Foo](p))
    print(w.greet(), call_greet(w))
    _ = f^
