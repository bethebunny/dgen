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
from std.builtin.variadics import TypeList
from std.ffi import _Global
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


def type_eq[A: AnyType, B: AnyType]() -> Bool:
    """Compile-time type identity."""
    return TypeList.of[Trait=AnyType, A]().contains[B]()


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
            comptime if D.accepts[T]():
                if key == type_id[D]():
                    return _static_vtable[D, downcast[T, D.Trait]]()
    return 0


# ===----------------------------------------------------------------------=== #
# Thunk helpers: turn an unbound method `T.m` into a receiver-erased thin fn.
#   vtable_for[T: Shape]() -> ShapeVT:  return ShapeVT(thunk[T.area])
# ===----------------------------------------------------------------------=== #


def thunk[T: AnyType, R: Movable, //, f: def(T) thin -> R](p: Opaque) -> R:
    return f(p.unsafe_bitcast[T]()[])


def thunk1[
    T: AnyType, A: AnyType, R: Movable, //, f: def(T, A) thin -> R
](p: Opaque, a: A) -> R:
    return f(p.unsafe_bitcast[T]()[], a)


def thunk_mut[
    T: AnyType, R: Movable, //, f: def(mut T) thin -> R
](p: Opaque) -> R:
    return f(p.unsafe_bitcast[T]()[])


def thunk_mut1[
    T: AnyType, A: AnyType, R: Movable, //, f: def(mut T, A) thin -> R
](p: Opaque, a: A) -> R:
    return f(p.unsafe_bitcast[T]()[], a)


# ===----------------------------------------------------------------------=== #
# Open-world conformance registry (optional)
#
# The universe is closed because Mojo has no globals or static initialisers.
# `_Global` (std.ffi, private) is a named, lazily-created process global
# backed by the runtime, which is enough for an explicit, Swift-style
# "conformance record" table: register[T, Adapter]() at startup, and
# `Header.witness` falls back to it. Not thread-safe; prototype only.
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _Conformance(ImplicitlyCopyable):
    var type_key: Int
    var adapter_key: Int
    var vtable: Int


def _empty_registry() -> List[_Conformance]:
    return []


comptime _registry = _Global["mojo_existential_registry", _empty_registry]


def _registry_ptr() -> Pointer[List[_Conformance], MutUntrackedOrigin]:
    try:
        return _registry.get_or_create_ptr()
    except:
        abort("existential: could not create conformance registry")


def register[T: AnyType, D: TraitAdapter]() where conforms_to(T, D.Trait):
    """Record at runtime that `T` conforms to `D.Trait`, with its vtable."""
    _registry_ptr()[].append(
        _Conformance(
            type_id[T](),
            type_id[D](),
            _static_vtable[D, downcast[T, D.Trait]](),
        )
    )


def unsafe_register_as[T: AnyType, D: TraitAdapter, vtable: D.VTable]():
    """Retroactive conformance: declare that `T` satisfies `D.Trait` via an
    explicitly supplied vtable, even though `T` never declared the trait.

    Mojo conformance is nominal and can't be added to existing types; witness
    tables are just function pointers, so they can. `vtable` is a compile-time
    value, placed in rodata like every other vtable. Build its entries with the
    thunk helpers over free functions, e.g. `DescribeVT(thunk[describe_square])`
    where `def describe_square(s: Square) -> String`.
    """
    _registry_ptr()[].append(
        _Conformance(
            type_id[T](),
            type_id[D](),
            Int(Pointer(to=global_constant[vtable]())),
        )
    )


def _registry_lookup(type_key: Int, adapter_key: Int) -> Int:
    for c in _registry_ptr()[]:
        if c.type_key == type_key and c.adapter_key == adapter_key:
            return c.vtable
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
        var addr = self.meta[].witness(type_id[D]())
        if addr == 0:
            addr = _registry_lookup(self.type_id(), type_id[D]())
        return addr

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

    @staticmethod
    def accepts[T: AnyType]() -> Bool:
        """Extra compile-time filter beyond `T: Trait`; default accepts all.

        Adapters that fix an associated member (e.g. `Element = Int`, like
        Rust's `dyn Container<Element = Int>`) override this so the universe
        never builds their vtable for a `T` with a different member.
        """
        return True


# ===----------------------------------------------------------------------=== #
# Any
# ===----------------------------------------------------------------------=== #


struct AnyBox[Base: type_of(AnyType), *Adapters: TraitAdapter](Erased, Movable):
    """An owning existential: some `T: Base`, erased. Use `Existential.Any`."""

    var _h: Header

    # NB: the bound is a bare `T: Self.Base`, with the rest in `where`.
    # - `T: Self.Base & Movable & Deinitable` silently drops `Self.Base`
    #   (Mojo 1.1), leaving Base unchecked.
    # - `where conforms_to(T, Self.Base)` is evaluated before `__extension`
    #   conformances are visible, so it rejects extension-conformed types.
    @implicit
    def __init__[
        T: Self.Base
    ](out self, var value: T) where conforms_to(T, Movable & Deinitable):
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

    def isa[T: AnyType](self) -> Bool:
        """Whether the erased type is exactly `T`."""
        return self._h.type_id() == type_id[T]()

    def take[
        T: Movable
    ](deinit self) raises TakeError[Self.Base, *Self.Adapters] -> T:
        """Move the erased value out as `T`, consuming the `Any`.

        On a type mismatch nothing is lost: the raised `TakeError` carries the
        `Any` back (like Rust's `Box<dyn Any>::downcast` returning `Err(self)`).
        """
        if self._h.type_id() != type_id[T]():
            raise TakeError(
                AnyBox[Self.Base, *Self.Adapters](unsafe_header=self._h),
                reflect[T].name(),
            )
        var p = self._h.data.unsafe_bitcast[T]()
        var value = p.unsafe_take_pointee()
        p.unsafe_free()  # the box only; T was moved out, not destroyed
        return value^


struct TakeError[Base: type_of(AnyType), *Adapters: TraitAdapter](
    Movable, Writable
):
    """Raised by `AnyBox.take` on a type mismatch; owns the original `Any`."""

    var value: AnyBox[Self.Base, *Self.Adapters]
    var expected: StaticString

    def __init__(
        out self,
        var value: AnyBox[Self.Base, *Self.Adapters],
        expected: StaticString,
    ):
        self.value = value^
        self.expected = expected

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "cannot take ", self.value.type_name(), " as ", self.expected
        )

    def into_value(deinit self) -> AnyBox[Self.Base, *Self.Adapters]:
        """Recover the `Any` that failed to be taken."""
        return self.value^


def _header_of[X: AnyType](ref x: X) -> Header:
    comptime assert conforms_to(X, Erased), "not an existential or witness"
    comptime assert size_of[X]() == size_of[Header]()
    return Pointer(to=x).unsafe_bitcast[Header]()[]


def trait_cast_via[
    D: TraitAdapter, X: AnyType
](ref x: X) raises -> ref[x] downcast[D, D.Trait]:
    """Universe-free trait cast: name the adapter instead of the trait.

    Succeeds if the erased type conforms via the static universe it was boxed
    with, or via a runtime `register[T, D]()` record.
    """
    var h = _header_of(x)
    if h.witness[D]() == 0:
        raise Error(
            String(h.type_name(), " does not conform to ", reflect[D].name())
        )
    return Pointer(to=x).unsafe_bitcast[downcast[D, D.Trait]]()[]


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

    comptime Extend[*More: TraitAdapter] = Existential[
        *TypeList._concat[Self.Adapters.values, More.values]()
    ]
    """A new, larger universe. Values boxed under `Self` don't see `More`."""

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
