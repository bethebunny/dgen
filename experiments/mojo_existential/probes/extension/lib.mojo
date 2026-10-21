# A "library" type that knows nothing about Greet.
@fieldwise_init
struct Point(Copyable, Writable):
    var x: Int
    var y: Int


@fieldwise_init
struct Wrapper[T: Copyable & Writable & Deinitable](Copyable, Writable):
    var inner: Self.T
