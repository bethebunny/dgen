# Without Any the borrow is in the type, so it can't escape: the return type
# can't name a local's origin -> 'cannot implicitly convert ... to None'.
trait Peek:
    def peek(self) -> Int:
        ...


@fieldwise_init
struct Borrower[o: ImmOrigin](Movable, Peek):
    var p: ImmPointer[Int, Self.o]

    def peek(self) -> Int:
        return self.p[]


# (a2) inferring the result type: the origin of a local can't be named in a signature
def make_plain():
    var xs: List[Int] = [1, 2, 3]
    var b = Borrower(Pointer(to=xs[1]))
    return b^


def main():
    make_plain()
