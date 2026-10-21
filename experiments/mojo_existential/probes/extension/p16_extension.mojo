# Probe: the undocumented `__extension` keyword (Mojo 1.1).
from lib import Point, Wrapper


trait Greet:
    def greet(self) -> String:
        ...


# (1) method + retroactive conformance on a type from another module
__extension Point(Greet):
    def greet(self) -> String:
        return String("hi from ", self.x, ",", self.y)


# (2) conditional conformance on a generic library type
# (2) conditional conformance in an extension is rejected:
#   __extension Wrapper(Greet where conforms_to(Self.T, Greet)):
#   -> error: 'where' clauses in conformance lists are only supported on structs
# (2b) unconditional conformance on a generic type, refining inside the body:
__extension Wrapper(Greet):
    def greet(self) -> String:
        comptime if conforms_to(Self.T, Greet):
            return String("wrapped: ", self.inner.greet())
        else:
            return String("wrapped non-greeter: ", self.inner)


# (3) a stdlib struct
__extension String(Greet):
    def greet(self) -> String:
        return String("string says ", self)


def use[G: Greet](g: G) -> String:
    return g.greet()


def main():
    print("(1)", use(Point(1, 2)))
    print("(2)", use(Wrapper(Point(3, 4))))
    print("(3)", use(String("yo")))
    print("conforms_to(Point, Greet):", conforms_to(Point, Greet))
    print(
        "conforms_to(Wrapper[String], Greet):",
        conforms_to(Wrapper[String], Greet),
    )
