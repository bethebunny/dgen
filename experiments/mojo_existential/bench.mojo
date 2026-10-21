# Rough dispatch-cost microbenchmark (not rigorous; single run, one machine).
#   static    List[Sq], monomorphic calls (inlinable)
#   fat       {data*, fnptr} pairs: the ideal one-indirect-call `dyn Trait`
#   any[]     List[Any[Shape]], x[].area(): meta.witness lookup + vtable call
#   any[] far same, but Shape is the 3rd adapter (witness scan does 3 compares)
from std.benchmark import keep
from std.time import perf_counter_ns

from existential import *


trait Shape:
    def area(self) -> Int:
        ...


trait A1:
    def a1(self) -> Int:
        ...


trait A2:
    def a2(self) -> Int:
        ...


@fieldwise_init
struct ShapeVT(ImplicitlyCopyable):
    var area: def(Opaque) thin -> Int


struct DynShape(Shape, TraitAdapter):
    comptime Trait = Shape
    comptime VTable = ShapeVT
    var _h: Header

    @staticmethod
    def vtable_for[T: Shape]() -> ShapeVT:
        return ShapeVT(thunk[T.area])

    def area(self) -> Int:
        return self._h.vtable[Self]().area(self._h.data)


@fieldwise_init
struct A1VT(ImplicitlyCopyable):
    var a1: def(Opaque) thin -> Int


struct DynA1(A1, TraitAdapter):
    comptime Trait = A1
    comptime VTable = A1VT
    var _h: Header

    @staticmethod
    def vtable_for[T: A1]() -> A1VT:
        return A1VT(thunk[T.a1])

    def a1(self) -> Int:
        return self._h.vtable[Self]().a1(self._h.data)


@fieldwise_init
struct A2VT(ImplicitlyCopyable):
    var a2: def(Opaque) thin -> Int


struct DynA2(A2, TraitAdapter):
    comptime Trait = A2
    comptime VTable = A2VT
    var _h: Header

    @staticmethod
    def vtable_for[T: A2]() -> A2VT:
        return A2VT(thunk[T.a2])

    def a2(self) -> Int:
        return self._h.vtable[Self]().a2(self._h.data)


@fieldwise_init
struct Sq(A1, A2, Copyable, Shape):
    var s: Int

    def area(self) -> Int:
        return self.s * self.s

    def a1(self) -> Int:
        return 1

    def a2(self) -> Int:
        return 2


@fieldwise_init
struct Tri(A1, A2, Copyable, Shape):
    var b: Int

    def area(self) -> Int:
        return self.b * 3

    def a1(self) -> Int:
        return 1

    def a2(self) -> Int:
        return 2


comptime near = Existential[DynShape, DynA1, DynA2]
comptime far = Existential[DynA1, DynA2, DynShape]

comptime N = 1024
comptime REPS = 10_000


def report(label: String, t0: Int, t1: Int, checksum: Int):
    var ns = Float64(t1 - t0) / Float64(N * REPS)
    print(label, ns, "ns/call  (checksum", checksum, ")")


def main() raises:
    var sqs = List[Sq]()
    var anys = List[near.Any[Shape]]()
    var fars = List[far.Any[Shape]]()
    var fat_data = List[Opaque]()
    var fat_fn = List[def(Opaque) thin -> Int]()
    for i in range(N):
        sqs.append(Sq(i))
        if i % 2 == 0:
            anys.append(Sq(i))
            fars.append(Sq(i))
        else:
            anys.append(Tri(i))
            fars.append(Tri(i))
    for i in range(N):
        var h = Pointer(to=anys[i]).unsafe_bitcast[Header]()[]
        fat_data.append(h.data)
        fat_fn.append(h.vtable[DynShape]().area)

    var acc = 0
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        for i in range(N):
            acc += sqs[i].area()
    keep(acc)
    report("static   ", t0, perf_counter_ns(), acc)

    acc = 0
    t0 = perf_counter_ns()
    for _ in range(REPS):
        for i in range(N):
            acc += fat_fn[i](fat_data[i])
    keep(acc)
    report("fat      ", t0, perf_counter_ns(), acc)

    acc = 0
    t0 = perf_counter_ns()
    for _ in range(REPS):
        for i in range(N):
            acc += anys[i][].area()
    keep(acc)
    report("any[]    ", t0, perf_counter_ns(), acc)

    acc = 0
    t0 = perf_counter_ns()
    for _ in range(REPS):
        for i in range(N):
            acc += fars[i][].area()
    keep(acc)
    report("any[] far", t0, perf_counter_ns(), acc)
