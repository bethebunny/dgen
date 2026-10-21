# Orphan case: imported struct + imported trait -> rejected (like Rust's orphan rule).
from ext_struct import Point
from ext_trait import Greet

__extension Point(Greet):
    def greet(self) -> String:
        return String("hi ", self.x)

def use[G: Greet](g: G) -> String:
    return g.greet()

def main():
    print("generic call:", use(Point(1)), "| conforms_to:", conforms_to(Point, Greet))
