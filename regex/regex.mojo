# Prototype file for the regex dialect.
# The idea is
#   1. hand-write the dialect types we'd want DGEN to generate
#   2. write the dialect file which should correspond to the generated code
#   3. write enough code generation to actually generate the desired code

trait Pattern:
    pass

comptime AnyPattern = Variant[Dot, Literal, Union, Concatenation, Repeated]

struct Dot(Pattern):
    pass

@fieldwise_init
struct Literal(Pattern):
    var pattern: String

@fieldwise_init
struct Union(Pattern):
    var lhs: OwnedPointer[AnyPattern]
    var rhs: OwnedPointer[AnyPattern]

@fieldwise_init
struct Concatenation(Pattern):
    var lhs: OwnedPointer[AnyPattern]
    var rhs: OwnedPointer[AnyPattern]

@fieldwise_init
struct Repeated(Pattern):
    var pattern: OwnedPointer[AnyPattern]
    var min: Int = 0
    var max: Optional[Int] = None

struct Match:
    pass

@fieldwise_init
struct MatchOp:
    var pattern: Value[Pattern]
    var string: Value[String]
