# Library code that boxes values. Its universe comes from a module the
# *application* provides on the include path: late binding by import.
from existential import *
from existential_universe import universe
from shapes import Shape

@fieldwise_init
struct Square(Movable, Shape):
    var s: Int
    def area(self) -> Int: return self.s * self.s
    def perimeter(self) -> Int: return 4 * self.s   # no Perimeter trait known here

def make_shape() -> universe.Any[Shape]:
    return Square(5)
