# Existential types for Mojo, built entirely in-language (Mojo 1.1).
#
#   dyn = Existential[DynA, DynB, ...]   a closed "universe" of trait adapters
#   dyn.Any[Base]     owning, type-erased box for some `T: Base`.
#                     16 bytes: {data*, meta*}. `meta` is a per-type singleton
#                     `TypeMeta` living in rodata (via `global_constant`).
#   dyn.Witness[Tr]   the per-trait adapter struct (conforms to `Tr`), seen
#                     only as "some type conforming to Tr".
#   dyn.trait_cast[Tr](x)  runtime-checked; returns `ref Witness[Tr]` that
#                     *aliases* `x` (an Any or another witness). No copy, no
#                     allocation: adapters are layout-identical to Any.
#   cast[T](x)        runtime-checked; returns `ref T` into the box.
#
# Each adapter method looks up a static vtable for (erased T, adapter) through
# `meta.witness`, then calls through it. Vtables are compile-time constants
# placed in rodata, so there is exactly one per (T, Trait) pair.

from std.builtin.globals import global_constant
from std.builtin.rebind import downcast
from std.memory.alloc import unsafe_alloc
from std.os import abort
from std.sys import size_of

comptime Opaque = MutOpaquePointer[MutUntrackedOrigin]
comptime StaticPtr[T: AnyType] = ImmPointer[T, ImmStaticOrigin]


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
    # The body differs per T, so identical-code-folding can't give two
    # instantiations the same address.
    return reflect[T].name()


def _fn_address(f: def() thin -> StaticString) -> Int:
    var g = f
    return Pointer(to=g).unsafe_bitcast[Int]()[]


def type_id[T: AnyType]() -> Int:
    """A process-unique Int per type: the address of `_type_tag[T]`."""
    return _fn_address(_type_tag[T])


# ===----------------------------------------------------------------------=== #
# Per-type metadata singleton
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct TypeMeta(ImplicitlyCopyable, Movable):
    """Static, per-(T, universe) metadata. One copy lives in rodata."""

    var type_tag: def() thin -> StaticString
    var destroy: def(Opaque) thin -> None
    var clone: def(Opaque) thin -> Int  # new box address, or 0 if not Copyable
    # Adapter key -> address of the static vtable for (T, adapter), or 0.
    var witness: def(Int) thin -> Int


def _destroy[T: Movable & Deinitable](data: Opaque):
    var p = data.unsafe_bitcast[T]()
    p.unsafe_deinit_pointee()
    p.unsafe_free()


def _clone[T: Movable & Deinitable](data: Opaque) -> Int:
    comptime if conforms_to(T, Copyable):
        var p = unsafe_alloc[T](1)
        p.unsafe_write(data.unsafe_bitcast[T]()[].copy())
        return Int(p)
    else:
        return 0


def _static_vtable[D: TraitAdapter, T: D.Trait]() -> Int:
    comptime vt = D.vtable_for[T]()
    return Int(Pointer(to=global_constant[vt]()))


def _witness[T: AnyType, *Adapters: TraitAdapter](key: Int) -> Int:
    comptime for i in range(Adapters.length):
        comptime D = Adapters[i]
        comptime if conforms_to(T, D.Trait):
            if key == type_id[D]():
                return _static_vtable[D, downcast[T, D.Trait]]()
    return 0


struct Header(ImplicitlyCopyable, Movable):
    """The 16-byte erased representation shared by `Any` and every adapter."""

    var data: Opaque
    var meta: StaticPtr[TypeMeta]

    def __init__(out self, data: Opaque, meta: StaticPtr[TypeMeta]):
        self.data = data
        self.meta = meta

    def type_id(self) -> Int:
        return _fn_address(self.meta[].type_tag)

    def type_name(self) -> StaticString:
        return self.meta[].type_tag()

    def witness[D: TraitAdapter](self) -> Int:
        return self.meta[].witness(type_id[D]())

    def vtable[D: TraitAdapter](self) -> ref[ImmStaticOrigin] D.VTable:
        var addr = self.witness[D]()
        if addr == 0:
            abort("existential: vtable requested for non-conforming type")
        return StaticPtr[D.VTable](unsafe_from_address=addr)[]


# ===----------------------------------------------------------------------=== #
# Trait adapters ("witness" structs)
# ===----------------------------------------------------------------------=== #


trait Erased:
    """Marker: a struct whose only field is a `Header`."""

    pass


trait TraitAdapter(Erased):
    """A struct that conforms to `Trait` by forwarding through a vtable.

    Must be layout-identical to `Header` (its only field is a `Header`), so
    that an `Any` can be reinterpreted in place as the adapter.
    """

    comptime Trait: type_of(AnyType)
    comptime VTable: ImplicitlyCopyable & Deinitable

    @staticmethod
    def vtable_for[T: Self.Trait]() -> Self.VTable:
        ...


# ===----------------------------------------------------------------------=== #
# Any
# ===----------------------------------------------------------------------=== #


struct AnyBox[Base: type_of(AnyType), *Adapters: TraitAdapter](Erased, Movable):
    """An owning existential: some `T: Base`, erased. Use `Existential.Any`."""

    var _h: Header

    # NB: not `T: Self.Base & Movable & Deinitable` -- composing a trait-typed
    # *parameter* silently drops it (Mojo 1.1), so Base would go unchecked.
    @implicit
    def __init__[
        T: Movable & Deinitable
    ](out self, var value: T) where conforms_to(T, Self.Base):
        comptime M = downcast[T, Movable & Deinitable]
        comptime meta = TypeMeta(
            _type_tag[M], _destroy[M], _clone[M], _witness[M, *Self.Adapters]
        )
        var p = unsafe_alloc[T](1)
        p.unsafe_write(value^)
        self._h = Header(
            p.unsafe_bitcast[NoneType](), Pointer(to=global_constant[meta]())
        )

    def __init__(out self, *, unsafe_header: Header):
        self._h = unsafe_header

    def __deinit__(deinit self):
        self._h.meta[].destroy(self._h.data)

    def type_name(self) -> StaticString:
        return self._h.type_name()

    def __getitem__(
        ref self,
    ) -> ref[self] Existential[*Self.Adapters].Witness[Self.Base]:
        """Infallible view as the base trait (guaranteed at construction)."""
        return Pointer(to=self).unsafe_bitcast[
            Existential[*Self.Adapters].Witness[Self.Base]
        ]()[]


def _header_of[X: AnyType](ref x: X) -> Header:
    comptime assert conforms_to(X, Erased), "not an existential or witness"
    comptime assert size_of[X]() == size_of[Header]()
    return Pointer(to=x).unsafe_bitcast[Header]()[]


def cast[T: AnyType, X: AnyType](ref x: X) raises -> ref[x] T:
    """Downcast an `Any` or witness to the concrete type `T`."""
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


# ===----------------------------------------------------------------------=== #
# The universe / namespace
# ===----------------------------------------------------------------------=== #


struct Existential[*Adapters: TraitAdapter]:
    """A closed universe of trait adapters, and the API bound to it."""

    @staticmethod
    def index_of[Tr: type_of(AnyType)]() -> Int:
        comptime for i in range(Self.Adapters.length):
            comptime if trait_eq[Self.Adapters[i].Trait, Tr]():
                return i
        return -1

    comptime AdapterFor[Tr: type_of(AnyType)]: TraitAdapter = Self.Adapters[
        Self.index_of[Tr]()
    ]

    comptime Any[Base: type_of(AnyType)] = AnyBox[Base, *Self.Adapters]

    comptime Witness[Tr: type_of(AnyType)] = downcast[Self.AdapterFor[Tr], Tr]

    @staticmethod
    def trait_cast[
        Tr: type_of(AnyType), X: AnyType
    ](ref x: X) raises -> ref[x] Self.Witness[Tr]:
        """View an `Any` or witness as `Tr`, if its erased type conforms."""
        comptime assert (
            Self.index_of[Tr]() >= 0
        ), "trait has no adapter in this universe"
        var h = _header_of(x)
        if h.witness[Self.AdapterFor[Tr]]() == 0:
            raise Error(
                String(
                    h.type_name(),
                    " does not conform to ",
                    reflect[Self.AdapterFor[Tr]].name(),
                )
            )
        return Pointer(to=x).unsafe_bitcast[Self.Witness[Tr]]()[]

    @staticmethod
    def clone[
        Base: type_of(AnyType), X: AnyType
    ](ref x: X) raises -> AnyBox[Base, *Self.Adapters]:
        """Copy the erased value into a new `Any[Base]` (if T is Copyable)."""
        var h = _header_of(x)
        # Base is checked against the *erased* type, not just the static one.
        _ = Self.trait_cast[Base](x)
        var addr = h.meta[].clone(h.data)
        if addr == 0:
            raise Error(String(h.type_name(), " is not Copyable"))
        return AnyBox[Base, *Self.Adapters](
            unsafe_header=Header(Opaque(unsafe_from_address=addr), h.meta)
        )
