# Probe: Any parameterised by a comptime TypeList *value*, and attempts to
# "append" to a module-level comptime list after the fact.
#
# (1) Works: a TypeList value parameter is interchangeable with the variadic
#     form (`AnyBox[Base, *list]`), and lists concatenate into new lists.
# (2) Redeclaring to append fails:
#       comptime L = TypeList.of[Int]()
#       comptime L = TypeList._concat[L.values, ...]()
#     -> error: invalid redefinition of 'L'
# (3) There is no mutating comptime append:
#       comptime _ = L.append[String]()
#     -> error: 'TypeList[Int]' value has no attribute 'append'
#     Comptime values are immutable and pure: every "append" is a new value
#     under a new name, so it yields a new, distinct Any type (== Extend).
# Side note: `comptime VTable = Int` fails to satisfy TraitAdapter's
# `vtable_for` signature check; a one-field struct works.
from std.builtin.variadics import TypeList
from existential import *
from std.sys import size_of

@fieldwise_init
struct VT(ImplicitlyCopyable):
    var x: Int

trait Ta:
    pass

trait Tb:
    pass

struct A(TraitAdapter):
    comptime Trait = Ta
    comptime VTable = VT
    var _h: Header
    @staticmethod
    def vtable_for[T: Ta]() -> VT:
        return VT(0)

struct B(TraitAdapter):
    comptime Trait = Tb
    comptime VTable = VT
    var _h: Header
    @staticmethod
    def vtable_for[T: Tb]() -> VT:
        return VT(1)

# (1) a list as a value parameter, unpacked into the existing variadic API
struct AnyL[Base: type_of(AnyType), adapters: TypeList[Trait=TraitAdapter, ...]]:
    comptime Box = AnyBox[Self.Base, *Self.adapters]

comptime L = TypeList.of[Trait=TraitAdapter, A]()
comptime L2 = TypeList._concat[L.values, TypeList.of[Trait=TraitAdapter, B]().values]()

@fieldwise_init
struct X(Movable, Ta, Tb):
    var v: Int

def main() raises:
    print("(1) len L:", len(L), " len L2:", len(L2))
    var x: AnyL[Ta, L2].Box = X(1)
    _ = Existential[*L2].trait_cast[Tb](x)
    print("(1) list-parameterised Any works; size", size_of[AnyL[Ta, L2].Box]())
