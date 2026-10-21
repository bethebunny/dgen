# Imported trait on a local struct: works (generic call accepts it).
@fieldwise_init
struct Point(Movable):
    var x: Int
from ext_trait import Greet

__extension Point(Greet):
    def greet(self: Point) -> String:
        return String("hi ", self.x)

def use[G: Greet](g: G) -> String:
    return g.greet()

def main():
    print("generic call:", use(Point(1)), "| conforms_to:", conforms_to(Point, Greet))
