# Generator edge cases: which traits are skipped (see gen/edge_cases.expected_skips).
# TypedRaises and SomeArg are emitted but fail to compile, because mojo doc
# drops the error type and the implicit Some[] parameter.
@fieldwise_init
struct MyErr(Movable, Writable):
    var code: Int
    def write_to(self, mut writer: Some[Writer]):
        writer.write("MyErr(", self.code, ")")

trait Parametric:
    def show[W: Writer](self, mut w: W): ...

trait ValueParam:
    def nth[n: Int](self) -> Int: ...

trait Static:
    @staticmethod
    def make() -> Int: ...

trait SelfArg:
    def same(self, other: Self) -> Bool: ...

trait Overloaded:
    def f(self, x: Int) -> Int: ...
    def f(self, x: String) -> Int: ...

trait RefReturn:
    def get(ref self) -> ref[self] Int: ...

trait Assoc:
    comptime Element: AnyType
    def size(self) -> Int: ...

trait StdParent(Writable):
    def tag(self) -> Int: ...

trait TypedRaises:
    def risky(self) raises MyErr -> Int: ...

trait Defaulted:
    def base(self) -> Int: ...
    def twice(self) -> Int:
        return 2 * self.base()

trait SomeArg:
    def take_some(self, x: Some[Writable]) -> Int: ...
