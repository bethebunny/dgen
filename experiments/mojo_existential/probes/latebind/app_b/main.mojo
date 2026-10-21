from existential import *
from existential_universe import universe
from factory import make_shape, Square
from shapes import Shape
from traits import Describe, DescribeVT, DynDescribe


# Retroactive conformance for a library type the app doesn't own.
def describe_square(sq: Square) -> String:
    return String("square with perimeter ", sq.perimeter())


def main() raises:
    var s = make_shape()  # boxed inside the library, under app B's universe
    print("app B area:", s[].area())
    try:
        _ = universe.trait_cast[Describe](s)
    except e:
        print("app B before:", e)  # Square never declared Describe
    unsafe_register_as[Square, DynDescribe, DescribeVT(thunk[describe_square])]()
    print("app B after:", universe.trait_cast[Describe](s).describe())
