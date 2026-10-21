# Existential types for Mojo, built entirely in-language.
#
#   Any[Base, U]      owning, type-erased box: {data*, manager}. 16 bytes.
#   Witness[Tr]       a *view* of the same 16 bytes reinterpreted as the
#                     per-trait adapter struct, which really conforms to Tr.
#   trait_cast[Tr]    checks (at runtime) that the erased type conforms to Tr
#                     and returns `ref Witness[Tr]` aliasing the box. No copy.
#   cast[T]           checks (at runtime) the erased type is exactly T and
#                     returns `ref T` into the box.
#
# The manager is one thin function pointer per (T, universe), in the style of
# libstdc++'s std::any `_Manager<T>::_S_manage`. It is simultaneously the
# runtime type identity and the dispatcher for destroy / name / conformance /
# vtable queries. Vtables are compile-time constants materialised on demand.
#
# A *universe* is the compile-time list of trait adapters that may be queried.
# It has to be closed: Mojo has no global variables / static initialisers, so
# there is no way to register (T, Trait) pairs from separate compilation sites.

from std.builtin.rebind import downcast
from std.memory.alloc import unsafe_alloc
from std.os import abort
from std.memory.maybe_uninit import MaybeUninit

comptime Opaque = MutOpaquePointer[MutUntrackedOrigin]


def trait_eq[a: type_of(AnyType), b: type_of(AnyType)]() -> Bool:
    """Compile-time trait identity."""
    return __mlir_attr[
        `#kgen.param.identical<`,
        `#kgen.type<`,
        +a,
        `> : !kgen.type`,
        `,`,
        `#kgen.type<`,
        +b,
        `> : !kgen.type`,
        `> : !kgen.scalar<bool>`,
    ]


# ===----------------------------------------------------------------------=== #
# Runtime type identity
# ===----------------------------------------------------------------------=== #


@no_inline
def _type_tag[T: AnyType]() -> StaticString:
    # Body differs per T (returns T's name), so identical-code-folding can't
    # merge two instantiations and give them the same address.
    return reflect[T].name()


def type_id[T: AnyType]() -> Int:
    """A process-unique Int per type: the address of `_type_tag[T]`."""
    var f: def() thin -> StaticString = _type_tag[T]
    return Pointer(to=f).unsafe_bitcast[Int]()[]


# ===----------------------------------------------------------------------=== #
# Manager protocol
# ===----------------------------------------------------------------------=== #

comptime OP_TYPE_ID = 0  # -> type_id[T]()
comptime OP_DESTROY = 1  # destroy the boxed T and free its storage
comptime OP_CONFORMS = 2  # key=type_id[Adapter] -> 1 if T: Adapter.Trait
comptime OP_VTABLE = 3  # key=type_id[Adapter] -> writes Adapter.VTable to out
comptime OP_NAME = 4  # writes StaticString to out

comptime Manager = def(Int, Int, Opaque, MutOpaquePointer[_]) thin -> Int


struct Header(ImplicitlyCopyable, Movable):
    """The 16-byte erased representation shared by `Any` and every adapter."""

    var data: Opaque
    var manager: Manager

    def __init__(out self, data: Opaque, manager: Manager):
        self.data = data
        self.manager = manager

    def type_id(self) -> Int:
        return self.manager(OP_TYPE_ID, 0, self.data, self.data)

    def type_name(self) -> StaticString:
        var out = MaybeUninit[StaticString]()
        _ = self.manager(
            OP_NAME,
            0,
            self.data,
            out.unsafe_ptr().unsafe_bitcast[NoneType](),
        )
        return out.unsafe_assume_init()

    def conforms[D: TraitAdapter](self) -> Bool:
        return self.manager(OP_CONFORMS, type_id[D](), self.data, self.data) != 0

    def vtable[D: TraitAdapter](self) -> D.VTable:
        var out = MaybeUninit[D.VTable]()
        var ok = self.manager(
            OP_VTABLE,
            type_id[D](),
            self.data,
            out.unsafe_ptr().unsafe_bitcast[NoneType](),
        )
        if not ok:
            abort("existential: vtable requested for non-conforming type")
        return out^.unsafe_assume_init()


# ===----------------------------------------------------------------------=== #
# Trait adapters ("witness" structs)
# ===----------------------------------------------------------------------=== #


trait TraitAdapter:
    """A struct that conforms to `Trait` by forwarding through a vtable.

    Must be layout-identical to `Header` (i.e. its only field is a `Header`),
    so that an `Any` can be reinterpreted in place as the adapter.
    """

    comptime Trait: type_of(AnyType)
    comptime VTable: Movable & Deinitable

    @staticmethod
    def vtable_for[T: Self.Trait]() -> Self.VTable:
        ...


# ===----------------------------------------------------------------------=== #
# The universe
# ===----------------------------------------------------------------------=== #


trait Universe:
    comptime AdapterFor[Tr: type_of(AnyType)]: TraitAdapter

    @staticmethod
    def index_of[Tr: type_of(AnyType)]() -> Int:
        ...


    @staticmethod
    def query[T: AnyType](op: Int, key: Int, out_: MutOpaquePointer[_]) -> Int:
        ...


struct Existential[*Adapters: TraitAdapter](Universe):
    """A closed universe of trait adapters."""

    @staticmethod
    def index_of[Tr: type_of(AnyType)]() -> Int:
        comptime for i in range(Self.Adapters.length):
            comptime if trait_eq[Self.Adapters[i].Trait, Tr]():
                return i
        return -1

    comptime AdapterFor[Tr: type_of(AnyType)]: TraitAdapter = Self.Adapters[
        Self.index_of[Tr]()
    ]

    @staticmethod
    def query[T: AnyType](op: Int, key: Int, out_: MutOpaquePointer[_]) -> Int:
        comptime for i in range(Self.Adapters.length):
            comptime D = Self.Adapters[i]
            if key == type_id[D]():
                comptime if conforms_to(T, D.Trait):
                    if op == OP_VTABLE:
                        out_.unsafe_bitcast[D.VTable]().unsafe_write(
                            D.vtable_for[downcast[T, D.Trait]]()
                        )
                    return 1
                else:
                    return 0
        return 0


def _manage[
    T: Movable & Deinitable, U: Universe
](op: Int, key: Int, data: Opaque, out_: MutOpaquePointer[_]) -> Int:
    if op == OP_TYPE_ID:
        return type_id[T]()
    if op == OP_DESTROY:
        var p = data.unsafe_bitcast[T]()
        p.unsafe_deinit_pointee()
        p.unsafe_free()
        return 1
    if op == OP_NAME:
        out_.unsafe_bitcast[StaticString]().unsafe_write(reflect[T].name())
        return 1
    return U.query[T](op, key, out_)


# ===----------------------------------------------------------------------=== #
# Any
# ===----------------------------------------------------------------------=== #


struct Any[Base: type_of(AnyType), U: Universe](Movable):
    """An owning existential: some `T: Base`, erased."""

    var _h: Header

    @implicit
    def __init__[T: Self.Base & Movable & Deinitable](out self, var value: T):
        var p = unsafe_alloc[T](1)
        p.unsafe_write(value^)
        self._h = Header(
            p.unsafe_bitcast[NoneType](), _manage[downcast[T, Movable & Deinitable], Self.U]
        )

    def __deinit__(deinit self):
        _ = self._h.manager(OP_DESTROY, 0, self._h.data, self._h.data)

    def type_name(self) -> StaticString:
        return self._h.type_name()


comptime Witness[Tr: type_of(AnyType), U: Universe] = downcast[
    U.AdapterFor[Tr], Tr
]


def _header_of[X: AnyType](ref x: X) -> Header:
    # Both `Any` and every adapter are a single `Header` field.
    return Pointer(to=x).unsafe_bitcast[Header]()[]


def trait_cast[
    Tr: type_of(AnyType), Base: type_of(AnyType), U: Universe
](ref x: Any[Base, U]) raises -> ref[x] Witness[Tr, U]:
    comptime assert U.index_of[Tr]() >= 0, "trait not in universe"
    var h = _header_of(x)
    if not h.conforms[U.AdapterFor[Tr]]():
        raise Error(
            String(h.type_name(), " does not conform to ", reflect[U.AdapterFor[Tr]].name())
        )
    return Pointer(to=x).unsafe_bitcast[Witness[Tr, U]]()[]


def cast[T: AnyType, X: AnyType](ref x: X) raises -> ref[x] T:
    var h = _header_of(x)
    if h.type_id() != type_id[T]():
        raise Error(
            String("cannot cast ", h.type_name(), " to ", reflect[T].name())
        )
    return (
        h.data.unsafe_bitcast[T]()
        .unsafe_mut_cast[origin_of(x).mut]()
        .unsafe_origin_cast[origin_of(x)]()[]
    )
