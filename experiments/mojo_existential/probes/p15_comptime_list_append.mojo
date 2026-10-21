# Probe: appending to a regular comptime List (not TypeList).
#
# (A)/(B) work: compile-time evaluation runs ordinary code, including mutation
#         of *local* lists, so you can *build* a comptime list with append.
# (C) mutating a named comptime value fails, at module scope or in a body:
#       comptime _r = xs.append(3)   /   xs.append(3)
#     -> error: invalid use of mutating method on rvalue of type 'List[Int]'
#     (a comptime name is an rvalue: each use is a fresh copy of the value)
# (D) a materialized runtime copy is mutable; the comptime value is unchanged.
# (E) redeclaring: comptime xs = [1, 2, 3] -> error: invalid redefinition of 'xs'
comptime xs: List[Int] = [1, 2]

def appended(var l: List[Int], x: Int) -> List[Int]:
    l.append(x)          # ordinary mutation, inside a function body
    return l^

comptime ys = appended(materialize[xs](), 3)    # (A) pure function at comptime

def build() -> List[Int]:
    var l = materialize[xs]()
    for i in range(3):   # (B) loops + appends, all evaluated by the comptime interpreter
        l.append(10 + i)
    return l^

comptime zs = build()

def main():
    print("(A) xs:", materialize[xs](), " ys:", materialize[ys]())
    print("(B) zs:", materialize[zs]())
    var local = materialize[xs]()
    local.append(99)     # (D) a runtime copy is mutable; xs is not
    print("(D) copy:", local, " xs still:", materialize[xs]())
